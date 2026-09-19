use cv_hal::DeviceId;
use cv_runtime::distributed::{ShmCoordinator, SHM_TOTAL_SIZE};
use cv_runtime::orchestrator::{scheduler, RuntimeRunner, WorkloadHint};
use cv_runtime::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn stress_test_gpu_cpu_interleaved() {
    // This test simulates a real-world pipeline where tasks hit both the CPU and the GPU
    // concurrently. We want to ensure that the global lock-free mechanisms and VRAM budgeting
    // do not deadlock when interleaved aggressively.
    let _s = scheduler().unwrap();
    let num_tasks = 1000;

    // We'll use the default CPU group for CPU tasks, and rely on `best_runner_gpu_wait` for GPU.

    let cpu_counter = Arc::new(AtomicUsize::new(0));
    let gpu_counter = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let barrier = Arc::new(Barrier::new(5)); // 4 workers + 1 main

    // Spawn 4 threads doing mixed workloads
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let b = barrier.clone();
            let cpu_c = cpu_counter.clone();
            let gpu_c = gpu_counter.clone();

            thread::spawn(move || {
                b.wait();
                for i in 0..(num_tasks / 4) {
                    if i % 2 == 0 {
                        // CPU bound task
                        if let Ok(Some(group)) = cv_runtime::orchestrator::scheduler()
                            .unwrap()
                            .get_best_group(cv_hal::BackendType::Cpu, WorkloadHint::Throughput)
                        {
                            let c = cpu_c.clone();
                            let _ = group.spawn(move || {
                                // Simulate CPU work
                                let mut x = 0;
                                for _ in 0..1000 {
                                    x += 1;
                                }
                                c.fetch_add(x - 999, Ordering::Relaxed);
                            });
                        }
                    } else {
                        // GPU bound task (or fallback if VRAM is full)
                        // We use run_safe with an idempotent closure
                        let (runner, _is_gpu) = cv_runtime::orchestrator::best_runner_gpu_wait_for(
                            WorkloadHint::Throughput,
                            10, // require 10MB
                            Some(Duration::from_millis(10)),
                        )
                        .unwrap_or_else(|_| (RuntimeRunner::Sync(DeviceId(0)), false));

                        let c = gpu_c.clone();
                        let _ = runner.run_safe(true, move || {
                            // Simulate GPU submission
                            c.fetch_add(1, Ordering::Relaxed);
                            Ok::<(), Error>(())
                        });
                    }
                }
            })
        })
        .collect();

    barrier.wait(); // Start!

    for h in handles {
        h.join().unwrap();
    }

    // Wait for async CPU tasks to clear their queues
    let mut spins = 0;
    while cpu_counter.load(Ordering::Relaxed) < (num_tasks / 2) {
        thread::sleep(Duration::from_millis(1));
        spins += 1;
        if spins > 5000 {
            panic!("CPU tasks did not complete");
        }
    }

    let elapsed = start.elapsed();
    println!("GPU/CPU Interleaved test completed in {:?}", elapsed);
    assert_eq!(cpu_counter.load(Ordering::Relaxed), num_tasks / 2);
    // run_safe runs synchronously for Sync runner, or via Group.spawn which we just wait for.
    // Actually run_safe uses run() which blocks on Sync but just spawns on Group.
    // If it spawned on a group, we need to wait for it.
    spins = 0;
    while gpu_counter.load(Ordering::Relaxed) < (num_tasks / 2) {
        thread::sleep(Duration::from_millis(1));
        spins += 1;
        if spins > 5000 {
            panic!("GPU tasks did not complete");
        }
    }
}

#[test]
fn test_async_process_interaction() {
    // Tests two completely isolated "processes" (simulated by using different ShmCoordinators
    // pointing to the same shared memory, acting as different PIDs via slot acquisition).
    let shm_name = format!("async_interact_{}", std::process::id());

    let coord1 = ShmCoordinator::new(&shm_name, SHM_TOTAL_SIZE).unwrap();
    // Give device 0 some memory
    coord1.init_device(0, 100).unwrap();

    let barrier = Arc::new(Barrier::new(3));
    let b1 = barrier.clone();
    let b2 = barrier.clone();

    let start = Instant::now();

    // "Process" 1: Grabs memory, holds it, then releases
    let name_p1 = shm_name.clone();
    let p1 = thread::spawn(move || {
        let local_coord = ShmCoordinator::new(&name_p1, SHM_TOTAL_SIZE).unwrap();
        b1.wait();

        local_coord.reserve_device(0, 80, 0).unwrap();
        println!("P1: Reserved 80MB. Sleeping...");
        thread::sleep(Duration::from_millis(100));
        println!("P1: Releasing memory.");
        local_coord.release_device(0).unwrap();
    });

    // "Process" 2: Tries to grab memory, gets blocked, is woken up by P1's release
    let name_p2 = shm_name.clone();
    let p2 = thread::spawn(move || {
        let local_coord = ShmCoordinator::new(&name_p2, SHM_TOTAL_SIZE).unwrap();
        b2.wait();

        // Brief sleep to ensure P1 gets the lock first
        thread::sleep(Duration::from_millis(10));

        println!("P2: Attempting to reserve 50MB (will block)...");
        let wait_start = Instant::now();
        // This should block on the futex until P1 releases
        local_coord
            .wait_for_device_memory(0, 50, Duration::from_secs(2))
            .unwrap();
        local_coord.reserve_device(0, 50, 0).unwrap();
        let waited = wait_start.elapsed();

        println!("P2: Woke up and reserved memory after {:?}", waited);
        // Ensure we actually waited the ~90ms left of P1's sleep
        assert!(
            waited >= Duration::from_millis(80),
            "P2 didn't wait long enough: {:?}",
            waited
        );

        local_coord.release_device(0).unwrap();
    });

    barrier.wait(); // sync start

    p1.join().unwrap();
    p2.join().unwrap();

    println!("Async interaction test passed in {:?}", start.elapsed());
}
