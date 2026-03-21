use cv_runtime::orchestrator::{scheduler, TaskPriority};
use cv_runtime::GroupPolicy;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[test]
fn test_disjoint_threads() {
    let s = scheduler().unwrap();

    let policy1 = GroupPolicy {
        priority: TaskPriority::Normal,
        allow_work_stealing: false, // Disallow stealing to ensure disjointness
        allow_dynamic_scaling: false,
    };
    let g1 = s.create_group("disjoint_1", 2, None, policy1).unwrap();

    let policy2 = GroupPolicy {
        priority: TaskPriority::Normal,
        allow_work_stealing: false, // Disallow stealing to ensure disjointness
        allow_dynamic_scaling: false,
    };
    let g2 = s.create_group("disjoint_2", 2, None, policy2).unwrap();

    let g1_threads = Arc::new(Mutex::new(HashSet::new()));
    let g2_threads = Arc::new(Mutex::new(HashSet::new()));

    let iterations = 100;
    let (tx, rx) = std::sync::mpsc::channel();

    for _ in 0..iterations {
        let g1_t = g1_threads.clone();
        let tx1 = tx.clone();
        let _ = g1.spawn(move || {
            g1_t.lock().unwrap().insert(thread::current().id());
            tx1.send(()).unwrap();
        });

        let g2_t = g2_threads.clone();
        let tx2 = tx.clone();
        let _ = g2.spawn(move || {
            g2_t.lock().unwrap().insert(thread::current().id());
            tx2.send(()).unwrap();
        });
    }

    // Wait for all 200 tasks to finish
    for _ in 0..(iterations * 2) {
        rx.recv_timeout(Duration::from_secs(5))
            .expect("Tasks did not finish in time");
    }

    let set1 = g1_threads.lock().unwrap();
    let set2 = g2_threads.lock().unwrap();

    // Verify that both groups executed tasks
    assert!(!set1.is_empty(), "Group 1 did not execute any tasks");
    assert!(!set2.is_empty(), "Group 2 did not execute any tasks");

    // Verify disjointness
    let intersection: HashSet<_> = set1.intersection(&set2).collect();
    assert!(
        intersection.is_empty(),
        "Threads overlapped! Intersection: {:?}",
        intersection
    );

    println!(
        "Disjoint threads verified. Group 1 threads: {}, Group 2 threads: {}",
        set1.len(),
        set2.len()
    );

    let _ = s.remove_group("disjoint_1");
    let _ = s.remove_group("disjoint_2");
}
