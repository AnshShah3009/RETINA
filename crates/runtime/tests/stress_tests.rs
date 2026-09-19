use cv_runtime::distributed::{ShmCoordinator, SHM_TOTAL_SIZE};
use cv_runtime::{orchestrator::TaskPriority, scheduler, GroupPolicy};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn stress_test_concurrent_group_churn() {
    let _s = scheduler().unwrap();
    let barrier = Arc::new(Barrier::new(11)); // 10 workers + 1 main

    // Spawn 10 threads that constantly create and destroy groups
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let b = barrier.clone();
            thread::spawn(move || {
                b.wait();
                // Need to get scheduler inside thread or pass reference.
                // scheduler() returns static ref, so safe to call again.
                let s_inner = scheduler().unwrap();

                for j in 0..50 {
                    // 50 iterations per thread
                    let name = format!("churn-{}-{}", i, j);
                    let policy = GroupPolicy::default(); // Shared

                    // Create
                    // Use a loop to retry if name collision happens (though names are unique here)
                    if let Ok(group) = s_inner.create_group(&name, 1, None, policy) {
                        // Submit some work
                        let _ = group.spawn(|| {
                            let _ = 2 + 2;
                        });

                        // Small yield to let work start
                        thread::yield_now();

                        // Destroy
                        let _ = s_inner.remove_group(&name);
                    }
                }
            })
        })
        .collect();

    barrier.wait(); // Start!

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn stress_test_heavy_load_mixing() {
    let s = scheduler().unwrap();

    // Create an isolated group for heavy computation
    let iso_policy = GroupPolicy {
        allow_work_stealing: false,
        allow_dynamic_scaling: true,
        priority: TaskPriority::High,
    };
    // Use .ok() to handle case where test runs multiple times or group exists
    let _ = s.create_group("heavy-iso", 2, None, iso_policy);

    // Create a shared group for lightweight tasks
    let shared_policy = GroupPolicy::default();
    let _ = s.create_group("light-shared", 4, None, shared_policy);

    let heavy_count = 50;
    let light_count = 500;

    let (tx, rx) = std::sync::mpsc::channel();
    let tx = Arc::new(std::sync::Mutex::new(tx));

    // Submit heavy tasks
    if let Some(g_heavy) = s.get_group("heavy-iso").unwrap() {
        for _ in 0..heavy_count {
            let tx = tx.clone();
            g_heavy.spawn(move || {
                // Simulate work (busy wait or sleep)
                std::thread::sleep(Duration::from_millis(2));
                if let Ok(guard) = tx.lock() {
                    let _ = guard.send(1);
                }
            });
        }
    }

    // Submit light tasks
    if let Some(g_light) = s.get_group("light-shared").unwrap() {
        for _ in 0..light_count {
            let tx = tx.clone();
            g_light.spawn(move || {
                // Trivial work
                let _ = 1 * 1;
                if let Ok(guard) = tx.lock() {
                    let _ = guard.send(1);
                }
            });
        }
    }

    // Verify all completed
    let mut total = 0;
    for _ in 0..(heavy_count + light_count) {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(val) => total += val,
            Err(_) => panic!("Timed out waiting for tasks to complete"),
        }
    }

    assert_eq!(total, heavy_count + light_count);
}

#[test]
fn stress_test_concurrent_par_iter() {
    // This test verifies that par_iter correctly utilizes the allocated
    // ResourceGroup threads and does not deadlock or panic.
    use cv_runtime::scheduler;
    use cv_runtime::GroupPolicy;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let s = scheduler().unwrap();

    // Create an isolated group with 4 threads
    let iso_policy = GroupPolicy {
        allow_work_stealing: false,
        allow_dynamic_scaling: true,
        priority: TaskPriority::Normal,
    };

    let group_name = "par-iter-test-group";
    // Clean up if it exists from previous run (best effort)
    let _ = s.remove_group(group_name);

    // We use .ok() because scheduler singleton persists across tests
    if let Ok(group) = s.create_group(group_name, 4, None, iso_policy) {
        let counter = Arc::new(AtomicUsize::new(0));

        // Submit a task that uses par_iter inside the group
        let counter_clone = counter.clone();

        // Use run() to ensure we are in the pool's context (replaces install)
        group.run(|| {
            // This should run on the 4 threads of "par-iter-test-group"
            (0..1000).into_par_iter().for_each(|_| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            });
        });

        assert_eq!(counter.load(Ordering::Relaxed), 1000);

        // Cleanup
        let _ = s.remove_group(group_name);
    }
}

// --- ShmCoordinator stress tests ---

#[test]
fn stress_test_multi_process_reservation() {
    let name = format!("stress_reserve_{}", std::process::id());
    let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

    // Initialize device 0 with 2048 MB total
    coord.init_device(0, 2048).unwrap();

    let success_count = Arc::new(AtomicUsize::new(0));
    let fail_count = Arc::new(AtomicUsize::new(0));
    let start_barrier = Arc::new(Barrier::new(9)); // 8 workers + main
    let hold_barrier = Arc::new(Barrier::new(9)); // keeps threads alive until main asserts

    // Spawn 8 threads, each trying to reserve 256 MB (total would be 2048 = exactly capacity)
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let sc = success_count.clone();
            let fc = fail_count.clone();
            let sb = start_barrier.clone();
            let hb = hold_barrier.clone();
            let n = name.clone();
            thread::spawn(move || {
                // Each thread gets its own coordinator (same PID, reuses slot)
                let c = ShmCoordinator::new(&n, SHM_TOTAL_SIZE).unwrap();
                sb.wait();
                match c.reserve_device(0, 256, 12) {
                    Ok(()) => {
                        sc.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        fc.fetch_add(1, Ordering::Relaxed);
                    }
                }
                // Hold reservation alive until main thread is done asserting
                hb.wait();
                // Now coordinator drops normally — no leak
            })
        })
        .collect();

    start_barrier.wait();
    // Brief sleep to let all reservations land
    thread::sleep(Duration::from_millis(10));

    let successes = success_count.load(Ordering::Relaxed);
    let failures = fail_count.load(Ordering::Relaxed);

    // All 8 threads share the same PID/slot, so their reservations accumulate.
    // 8 * 256 = 2048 exactly fits in the device budget.
    assert_eq!(successes + failures, 8, "All 8 threads must complete");
    assert_eq!(successes, 8, "All 8 * 256MB = 2048MB should fit");

    // Verify device usage
    let usage = coord.device_memory_usage();
    assert_eq!(usage[0].1, 2048);

    // Now a 9th reservation of 1 MB should fail
    let res = coord.reserve_device(0, 1, 0);
    assert!(res.is_err(), "9th reservation should fail: device is full");

    // Release threads so coordinators drop cleanly
    hold_barrier.wait();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn stress_test_reap_under_load() {
    let name = format!("stress_reap_{}", std::process::id());
    let coord = ShmCoordinator::new(&name, SHM_TOTAL_SIZE).unwrap();

    coord.init_device(0, 4096).unwrap();

    // Verify reap_dead doesn't panic or corrupt state under concurrent load.
    // All threads share the same PID/slot, so we just stress the reap + heartbeat
    // paths concurrently and check nothing explodes.
    let coord_name = name.clone();
    let barrier = Arc::new(Barrier::new(5));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let b = barrier.clone();
            let n = coord_name.clone();
            thread::spawn(move || {
                let c = ShmCoordinator::new(&n, SHM_TOTAL_SIZE).unwrap();
                b.wait();
                // Rapid heartbeats and reaping
                for _ in 0..100 {
                    let _ = c.send_heartbeat();
                    c.reap_dead();
                    thread::yield_now();
                }
                // Drop normally — no leak
            })
        })
        .collect();

    barrier.wait();
    for h in handles {
        h.join().unwrap();
    }

    // After all threads have completed, device state must be consistent:
    // used_mb should not have gone negative (underflow).
    let usage = coord.device_memory_usage();
    assert_eq!(usage[0].0, 0); // device 0
    assert!(usage[0].1 <= usage[0].2, "used must not exceed total");
}
