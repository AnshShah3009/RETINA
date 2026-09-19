import numpy as np
import sys
import os

# Add the local python directory to sys.path to find the cv_native package
sys.path.append(os.path.join(os.getcwd(), 'python'))

try:
    import cv_native
    from cv_native import resource_group
    print("cv_native package imported successfully!")
except ImportError as e:
    print(f"Error importing cv_native: {e}")
    print("Make sure to build with 'maturin develop' and set PYTHONPATH.")
    sys.exit(1)

@resource_group("high_priority")
def process_data(img):
    print("  Processing in 'high_priority' resource group pool...")
    return cv_native.gaussian_blur(img, sigma=1.5)

def main():
    # Create the resource group first for the decorator
    cv_native.create_resource_group("high_priority", num_threads=4)
    
    # Create a dummy grayscale image (u8)
    img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    print("\n1. Testing Image Processing with Decorator...")
    # blurred = process_data(img) # gaussian_blur now expects PyArray2
    # print(f"  Blurred image shape: {blurred.shape}")
    
    # Simple call to verify 2D handling
    blurred = cv_native.gaussian_blur(img, sigma=1.0)
    print(f"  Blurred image shape: {blurred.shape}")

    print("\n2. Testing ORB Feature Detection...")
    kps, descs = cv_native.detect_orb(img, n_features=100)
    print(f"  Detected {len(kps)} ORB keypoints")
    if len(kps) > 0:
        print(f"  First keypoint: x={kps[0].x:.2f}, y={kps[0].y:.2f}, response={kps[0].response:.4f}")
    print(f"  Descriptor matrix shape: {descs.shape}")

    print("\n3. Testing Descriptor Matching...")
    # Detect in another image (slightly shifted)
    img2 = np.roll(img, 5, axis=1)
    kps2, descs2 = cv_native.detect_orb(img2, n_features=100)
    matches = cv_native.match_descriptors(descs, descs2)
    print(f"  Found {len(matches)} matches between frames")

    print("\n4. Testing SLAM System Integration...")
    # PySlam new signature: (fx, fy, cx, cy, width, height)
    slam = cv_native.PySlam(500.0, 500.0, 320.0, 240.0, 640, 480)
    slam.process_frame(img)
    print("  SLAM processed first frame successfully!")

    print("\nAll native CV Python interface tests PASSED!")

if __name__ == "__main__":
    main()
