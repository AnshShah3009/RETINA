use cv_runtime::{orchestrator::TaskPriority, scheduler, GroupPolicy};
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
                        group.spawn(|| {
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
