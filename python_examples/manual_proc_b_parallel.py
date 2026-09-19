import cv_native as retina
import time
import os

def main():
    # Must match the shared memory name from Process A (Parallel)
    os.environ["CV_RUNTIME_SHM"] = "parallel_stress_test"

    print("--- GPU Process B (Parallel) ---")
    
    # We only request 40MB. Since A is holding 30MB, and capacity is 100MB,
    # 30 + 40 = 70MB <= 100MB. This should NOT block.
    request_mb = 40
    print(f"Attempting to reserve {request_mb}MB of VRAM...")
    print("If Process A is running (holding 30MB), this should succeed instantly without waiting!")
    
    start = time.time()
    try:
        # Wait up to 20 seconds for memory (will return instantly because it fits)
        retina.PyRuntime.wait_for_gpu(0, request_mb, 20000) 
        
        # Reserve the memory
        retina.PyRuntime.reserve_device(0, request_mb)
        waited = time.time() - start
        
        print(f"Success! Reserved {request_mb}MB of VRAM instantly (wait time: {waited:.4f} seconds).")
        print("Both processes are now holding VRAM simultaneously in parallel.")
        print("Holding for 5 seconds to demonstrate concurrent usage...")
        
        for i in range(5):
            print(f"Process B active... {5-i}s remaining")
            time.sleep(1)
        
        print(f"Releasing {request_mb}MB of VRAM.")
        retina.PyRuntime.release_device(0)
        print("Process B finished.")
        
    except Exception as e:
        print(f"Failed or Timed out: {e}")

if __name__ == "__main__":
    main()
