import cv_native as retina
import time
import os

def main():
    os.environ["CV_RUNTIME_SHM"] = "manual_stress_test"

    print("--- GPU Process A ---")
    
    # We must explicitly query the devices once so the Rust backend
    # populates the shared memory coordinator with the detected hardware limits.
    num_devs = retina.PyRuntime.get_num_devices()
    print(f"Detected {num_devs} logical devices.")

    # We use Strict mode so we wait/fail instead of falling back to CPU
    retina.PyRuntime.set_execution_mode(retina.PyExecutionMode.Strict)
    
    # Manually initialize memory limit for the coordinator
    retina.PyRuntime.mock_init_device(0, 100) # Device 0 now has 100MB
    
    print("Attempting to reserve 80MB of VRAM...")
    try:
        retina.PyRuntime.reserve_device(0, 80)
        print("Success! Holding 80MB for 15 seconds...")
        print(">>> NOW RUN Process B in the other window! <<<")
        
        for i in range(15):
            print(f"Holding... {15-i}s remaining")
            time.sleep(1)
            
        print("Releasing 80MB of VRAM.")
        retina.PyRuntime.release_device(0)
        print("Process A finished.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
