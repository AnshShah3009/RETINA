import subprocess
import time
import sys

def main():
    print("Launching two concurrent compute-heavy Python scripts...")
    
    # We will launch one in normal mode and one in adaptive_basic mode
    # Both should use the SHM coordinator automatically due to retina's internal init
    
    # Enable shared memory coordinator globally
    env = dict(sys.modules['os'].environ)
    env["CV_RUNTIME_SHM"] = "concurrent_stress_test"
    
    cmd1 = [sys.executable, "stress_test_orchestrator.py", "--mode", "normal", "--iterations", "50"]
    cmd2 = [sys.executable, "stress_test_orchestrator.py", "--mode", "adaptive_basic", "--iterations", "50"]
    
    p1 = subprocess.Popen(cmd1, env=env)
    p2 = subprocess.Popen(cmd2, env=env)
    
    print(f"Launched processes with PIDs: {p1.pid} and {p2.pid}")
    
    # Wait for both to finish
    p1.wait()
    p2.wait()
    
    print("Both processes finished.")

if __name__ == "__main__":
    main()
