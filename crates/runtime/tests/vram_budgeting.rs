use cv_runtime::distributed::{ShmCoordinator, SHM_TOTAL_SIZE};

#[test]
fn test_vram_budgeting_automatic_integration() {
    let shm_name = format!("test_vram_auto_{}", std::process::id());
    std::env::set_var("CV_RUNTIME_SHM", &shm_name);

    let coord =
        ShmCoordinator::new(&shm_name, SHM_TOTAL_SIZE).expect("Failed to create coordinator");
    coord.init_device(0, 1024).expect("Failed to init device 0");

    // Check initial usage
    let usage = coord.device_memory_usage();
    let initial_used = usage
        .iter()
        .find(|(idx, _, _)| *idx == 0)
        .map(|(_, used, _)| *used)
        .unwrap_or(0);
    assert_eq!(initial_used, 0, "Initial usage should be 0");

    // Reserve and release via coordinator directly
    coord.reserve_device(0, 4, 0).expect("Failed to reserve 4MB");

    let usage_after_reserve = coord.device_memory_usage();
    let used_after_reserve = usage_after_reserve
        .iter()
        .find(|(idx, _, _)| *idx == 0)
        .map(|(_, used, _)| *used)
        .unwrap_or(0);
    assert_eq!(
        used_after_reserve, 4,
        "Should reflect 4MB reservation"
    );

    coord.release_device(0).expect("Failed to release");

    let usage_after_release = coord.device_memory_usage();
    let used_after_release = usage_after_release
        .iter()
        .find(|(idx, _, _)| *idx == 0)
        .map(|(_, used, _)| *used)
        .unwrap_or(0);
    assert_eq!(
        used_after_release, 0,
        "VRAM should be released after release_device"
    );
}
