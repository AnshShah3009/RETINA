import cv_native as retina
import time
import os

def main():
    # Use a unique shared memory space for this parallel test
    os.environ["CV_RUNTIME_SHM"] = "parallel_stress_test"

    print("--- GPU Process A (Parallel) ---")
    
    # Initialize the device to have 100MB of total capacity
    retina.PyRuntime.mock_init_device(0, 100) 
    
    # We only reserve 30MB this time, leaving 70MB free
    request_mb = 30
    print(f"Attempting to reserve {request_mb}MB out of 100MB...")
    
    try:
        retina.PyRuntime.reserve_device(0, request_mb)
        print("Success! Holding 30MB for 15 seconds...")
        print(">>> NOW RUN Process B (Parallel) in the other window! <<<")
        
        for i in range(15):
            print(f"Holding... {15-i}s remaining")
            time.sleep(1)
            
        print("Releasing 30MB of VRAM.")
        retina.PyRuntime.release_device(0)
        print("Process A finished.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
