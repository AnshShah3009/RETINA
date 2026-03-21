use cv_hal::DeviceId;
use cv_runtime::distributed::{ShmCoordinator, SHM_TOTAL_SIZE};
use cv_runtime::orchestrator::{
    scheduler, set_execution_mode, AdaptiveLevel, ExecutionMode, RuntimeRunner, TaskPriority,
    WorkloadHint,
};
use cv_runtime::{Error, GroupPolicy};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// Global lock to prevent modes from interfering across tests
static PERF_TEST_LOCK: Mutex<()> = parking_lot::const_mutex(());

#[test]
fn perf_test_orchestrator_throughput() {
    let _guard = PERF_TEST_LOCK.lock();
    set_execution_mode(ExecutionMode::Normal);

    let s = scheduler().unwrap();
    let num_tasks = 100_000;

    // Create multiple groups
    for i in 0..4 {
        let name = format!("perf_group_{}", i);
        let _ = s.remove_group(&name);
        let policy = GroupPolicy {
            priority: TaskPriority::Normal,
            allow_work_stealing: true,
            allow_dynamic_scaling: false,
        };
        s.create_group(&name, 2, None, policy).unwrap();
    }

    let counter = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let barrier = Arc::new(Barrier::new(5));

    // Submit tasks from 4 threads concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let s_clone = scheduler().unwrap();
            let counter_clone = counter.clone();
            let b = barrier.clone();

            thread::spawn(move || {
                b.wait();
                for _ in 0..(num_tasks / 4) {
                    // Find best group dynamically
                    if let Ok(Some(group)) =
                        s_clone.get_best_group(cv_hal::BackendType::Cpu, WorkloadHint::Throughput)
                    {
                        let c = counter_clone.clone();
                        let _ = group.spawn(move || {
                            c.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                }
            })
        })
        .collect();

    barrier.wait();
    for h in handles {
        h.join().unwrap();
    }

    // Wait for all tasks to complete
    let mut spins = 0;
    while counter.load(Ordering::Relaxed) < num_tasks {
        thread::sleep(Duration::from_millis(1));
        spins += 1;
        if spins > 10000 {
            panic!("Tasks did not complete in time");
        }
    }
    let elapsed = start.elapsed();

    println!(
        "perf_test_orchestrator_throughput: Scheduled and executed {} tasks in {:?}",
        num_tasks, elapsed
    );
    assert_eq!(counter.load(Ordering::Relaxed), num_tasks);

    // Cleanup
    for i in 0..4 {
        let _ = s.remove_group(&format!("perf_group_{}", i));
    }
}

#[test]
fn perf_test_adaptive_mode_overhead() {
    let _guard = PERF_TEST_LOCK.lock();
    let _s = scheduler().unwrap();

    let iterations = 10_000;
    let runner = RuntimeRunner::Sync(DeviceId(999)); // Non-existent/mock device to force fallback

    let run_benchmark = |mode: ExecutionMode, mode_name: &str| -> Duration {
        set_execution_mode(mode);
        let counter = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();

        for _ in 0..iterations {
            let c = counter.clone();
            let _: Result<(), Error> = runner.run_safe(true, || {
                c.fetch_add(1, Ordering::Relaxed);
                Err(Error::RuntimeError("Simulated Failure".into()))
            });
        }
        let elapsed = start.elapsed();

        // In Strict mode, task runs once per iteration.
        // In Adaptive mode, task runs twice per iteration (once on failing GPU, once on CPU fallback).
        let expected = if mode == ExecutionMode::Strict {
            iterations
        } else {
            iterations * 2
        };
        assert_eq!(counter.load(Ordering::Relaxed), expected);

        println!(
            "{}: {} iterations took {:?}",
            mode_name, iterations, elapsed
        );
        elapsed
    };

    let strict_dur = run_benchmark(ExecutionMode::Strict, "Strict Mode");
    let adaptive_dur = run_benchmark(
        ExecutionMode::Adaptive(AdaptiveLevel::Basic),
        "Adaptive Mode (Basic)",
    );

    // Adaptive overhead should be reasonable (e.g. less than 5x strict due to double execution and registry lookup)
    assert!(
        adaptive_dur.as_micros() <= strict_dur.as_micros() * 10,
        "Adaptive overhead is too high"
    );
}

#[test]
fn perf_test_vram_wait_latency() {
    let _guard = PERF_TEST_LOCK.lock();
    let shm_name = format!("perf_vram_{}", std::process::id());
    let coord = ShmCoordinator::new(&shm_name, SHM_TOTAL_SIZE).unwrap();

    coord.init_device(0, 100).unwrap();
    coord.reserve_device(0, 100, 0).unwrap(); // Fully allocate device 0

    let start = Instant::now();
    let wait_handle = {
        let name = shm_name.clone();
        thread::spawn(move || {
            let local_coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();
            let wait_start = Instant::now();
            // Wait for 50MB to become free
            let res = local_coord.wait_for_device_memory(0, 50, Duration::from_secs(5));
            let waited = wait_start.elapsed();
            assert!(res.is_ok());
            waited
        })
    };

    // Give the wait thread a moment to block on the futex
    thread::sleep(Duration::from_millis(50));

    // Release memory, which should trigger an instant futex wake_all
    let release_time = Instant::now();
    coord.release_device(0).unwrap();

    let waited_dur = wait_handle.join().unwrap();
    let latency = waited_dur.saturating_sub(release_time.duration_since(start));

    println!(
        "perf_test_vram_wait_latency: Futex wake latency approx {:?}",
        latency
    );

    // The futex wake should be virtually instantaneous (< 10ms), unlike the old 50ms polling loop.
    assert!(
        latency < Duration::from_millis(20),
        "Futex wake latency too high: {:?}",
        latency
    );
}

#[test]
fn edge_case_rapid_group_reconfiguration() {
    let _guard = PERF_TEST_LOCK.lock();
    set_execution_mode(ExecutionMode::Adaptive(AdaptiveLevel::Aggressive));

    let s = scheduler().unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

    // Thread 1: Rapidly creates and destroys groups
    let r1 = running.clone();
    let b1 = barrier.clone();
    let h1 = thread::spawn(move || {
        b1.wait();
        let s_inner = scheduler().unwrap();
        let mut i = 0;
        while r1.load(Ordering::Relaxed) {
            let name = format!("edge_group_{}", i % 5);
            let policy = GroupPolicy::default();
            let _ = s_inner.create_group(&name, 1, None, policy);
            thread::yield_now();
            let _ = s_inner.remove_group(&name);
            i += 1;
        }
    });

    // Thread 2: Continuously queries best group and spawns tasks
    let r2 = running.clone();
    let b2 = barrier.clone();
    let h2 = thread::spawn(move || {
        b2.wait();
        let s_inner = scheduler().unwrap();
        while r2.load(Ordering::Relaxed) {
            if let Ok(Some(group)) =
                s_inner.get_best_group(cv_hal::BackendType::Cpu, WorkloadHint::Default)
            {
                let _ = group.spawn(|| {
                    // Trivial work
                    let _ = 2 * 2;
                });
            }
        }
    });

    barrier.wait();

    // Let chaos run for 500ms
    thread::sleep(Duration::from_millis(500));
    running.store(false, Ordering::Relaxed);

    h1.join().unwrap();
    h2.join().unwrap();

    // Cleanup any leftovers
    for i in 0..5 {
        let _ = s.remove_group(&format!("edge_group_{}", i));
    }
}
