//! Test to verify GPU initialization and detection.

use cv_hal::context::ComputeContext;
use cv_hal::gpu::GpuContext;

#[test]
fn test_gpu_init() {
    println!("\n=== GPU Initialization Test ===");

    // Try to initialize the global GPU context
    match GpuContext::new() {
        Ok(ctx) => {
            println!("✓ GPU Context initialized successfully!");
            println!("  Backend: {:?}", ctx.backend_type());
            println!("  Device ID: {:?}", ctx.device_id());
        }
        Err(e) => {
            println!("✗ Failed to initialize GPU: {}", e);
        }
    }
}
