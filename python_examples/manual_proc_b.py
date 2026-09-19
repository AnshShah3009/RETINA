import cv_native as retina
import time
import os
import sys

def main():
    # Must match the shared memory name from Process A
    os.environ["CV_RUNTIME_SHM"] = "manual_stress_test"

    print("--- GPU Process B ---")
    print("Initializing retina runtime...")
    retina.PyRuntime.set_execution_mode(retina.PyExecutionMode.Strict)
    
    print("Attempting to reserve 50MB of VRAM...")
    print("If Process A is running and holding 80MB, this should block and wait!")
    
    start = time.time()
    try:
        # Wait up to 20 seconds for 50MB to become free
        print(f"Waiting for VRAM... (Timeout: 20s)")
        retina.PyRuntime.wait_for_gpu(0, 50, 20000) 
        
        # Once it wakes up (because A released), actually reserve it
        retina.PyRuntime.reserve_device(0, 50)
        waited = time.time() - start
        
        print(f"Success! Reserved 50MB of VRAM after waiting {waited:.2f} seconds.")
        print("Holding for 2 seconds...")
        time.sleep(2)
        
        print("Releasing 50MB of VRAM.")
        retina.PyRuntime.release_device(0)
        print("Process B finished.")
        
    except Exception as e:
        print(f"Failed or Timed out: {e}")

if __name__ == "__main__":
    main()
