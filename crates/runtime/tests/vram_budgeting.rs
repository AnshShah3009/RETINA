use cv_hal::DeviceId;
use cv_runtime::distributed::{ShmCoordinator, SHM_TOTAL_SIZE};
use cv_runtime::memory::UnifiedBuffer;
use cv_runtime::scheduler;

#[test]
fn test_vram_budgeting_automatic_integration() {
    let shm_name = format!("test_vram_auto_{}", std::process::id());
    std::env::set_var("CV_RUNTIME_SHM", &shm_name);

    // Initialize the scheduler (this will pick up the SHM env var)
    let _s = scheduler().expect("Failed to initialize scheduler");

    // Initialize device 0 in the coordinator
    // We assume device 0 is a GPU if one exists
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

    // Create a UnifiedBuffer with auto-reserve.
    {
        let mut buf: UnifiedBuffer<f32> = UnifiedBuffer::new(1024 * 1024).with_auto_reserve(); // 4MB

        // Sync to device (assuming device 0 is the GPU)
        let _ = buf.sync_to_device(DeviceId(0));

        let usage_after_sync = coord.device_memory_usage();
        let used_after_sync = usage_after_sync
            .iter()
            .find(|(idx, _, _)| *idx == 0)
            .map(|(_, used, _)| *used)
            .unwrap_or(0);

        // EXPECTATION: Should be 4 (MB).
        assert_eq!(
            used_after_sync, 4,
            "UnifiedBuffer should automatically reserve 4MB of VRAM"
        );
    }

    // After dropping, it should return to 0
    let usage_after_drop = coord.device_memory_usage();
    let used_after_drop = usage_after_drop
        .iter()
        .find(|(idx, _, _)| *idx == 0)
        .map(|(_, used, _)| *used)
        .unwrap_or(0);
    assert_eq!(
        used_after_drop, 0,
        "VRAM should be released after UnifiedBuffer is dropped"
    );
}
