//! Test to verify GPU detection and report fallback behavior.

use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::best_runner;

#[test]
fn test_device_detection_report() {
    println!("\n========================================");
    println!("       GPU DETECTION TEST REPORT");
    println!("========================================");

    // Try to get the best runner
    match best_runner() {
        Ok(runner) => {
            println!("\n📱 Best Runner Device ID: {:?}", runner.device_id());

            // Try to get the compute device
            match runner.try_device() {
                Ok(compute_device) => {
                    println!("\n📊 Compute Device Type: ");

                    match compute_device {
                        ComputeDevice::Gpu(_) => {
                            println!("   ✓ GPU is AVAILABLE and selected!");
                            println!("   → Tests should run on GPU");
                        }
                        ComputeDevice::Mlx(_) => {
                            println!("   ✓ MLX (Apple Silicon) is AVAILABLE and selected!");
                            println!("   → Tests should run on MLX");
                        }
                        ComputeDevice::Cpu(_) => {
                            println!("   ⚠ CPU only - GPU NOT AVAILABLE");
                            println!("   → Tests will fallback to CPU");
                        }
                    }

                    println!("\n💡 To enable GPU:");
                    println!("   - Ensure NVIDIA/AMD GPU with drivers installed");
                    println!("   - Or use Apple Silicon Mac");
                    println!("   - Or run with VK_ICD_FILENAMES for Vulkan");
                }
                Err(e) => {
                    println!("\n⚠ Failed to get compute device: {}", e);
                    println!("   → Tests will fallback to CPU");
                }
            }
        }
        Err(e) => {
            println!("\n⚠ Failed to get best runner: {}", e);
            println!("   → Tests will use default CPU device");
        }
    }

    println!("\n========================================");
    println!("       FALLBACK TESTING");
    println!("========================================");
    println!("\nTo monitor GPU->CPU fallbacks during tests:");
    println!("  RUST_LOG=warn cargo test 2>&1 | grep -i warn");
    println!("\nIf you see warnings like:");
    println!("  'GPU not available, falling back to CPU'");
    println!("  Then the test is NOT running on GPU.");
    println!("\n========================================\n");
}
