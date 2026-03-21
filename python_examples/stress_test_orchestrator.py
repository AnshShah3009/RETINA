import cv2
import numpy as np
import time
import sys
import argparse
import cv_native as retina

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="normal", choices=["strict", "normal", "adaptive_basic", "adaptive_aggressive"])
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    # Set execution mode
    if args.mode == "strict":
        retina.PyRuntime.set_execution_mode(retina.PyExecutionMode.Strict)
    elif args.mode == "normal":
        retina.PyRuntime.set_execution_mode(retina.PyExecutionMode.Normal)
    elif args.mode == "adaptive_basic":
        retina.PyRuntime.set_execution_mode(retina.PyExecutionMode.AdaptiveBasic)
    elif args.mode == "adaptive_aggressive":
        retina.PyRuntime.set_execution_mode(retina.PyExecutionMode.AdaptiveAggressive)

    print(f"[{time.time()}] Starting stress test in {args.mode} mode for {args.iterations} iterations...")
    
    # Generate large random point cloud data (N x 3)
    num_points = 50000
    points = np.random.rand(num_points, 3).astype(np.float32)
    
    start_time = time.time()
    
    for i in range(args.iterations):
        # Perform normal estimation (this exercises the orchestrator, GPU/CPU dispatch)
        _ = retina.estimate_normals_np(points, 15)
        
        if i % 10 == 0:
            print(f"[{time.time()}] Process {sys.argv[0]}: Iteration {i}/{args.iterations}")
            
    end_time = time.time()
    print(f"[{time.time()}] Finished {args.iterations} iterations in {end_time - start_time:.2f} seconds.")
    
    # Print dispatch log to see if any fallbacks happened
    log = retina.PyRuntime.get_dispatch_log()
    fallbacks = sum(1 for e in log if e.get("fallback") == "true")
    print(f"Total fallbacks to CPU: {fallbacks}")

if __name__ == "__main__":
    main()

