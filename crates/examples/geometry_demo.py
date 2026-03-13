import numpy as np
import cv_native
import time

def test_iou():
    print("Testing standalone Rect IoU...")
    r1 = cv_native.PyRect(0, 0, 10, 10)
    r2 = cv_native.PyRect(5, 5, 10, 10)
    
    val = r1.iou(r2)
    print(f"  IoU(r1, r2): {val:.4f} (Expected: 0.1429)")
    assert abs(val - 0.142857) < 1e-4

def test_vectorized_iou():
    print("\nTesting Vectorized Rect IoU...")
    # Create random boxes (N, 4)
    N = 1000
    M = 500
    boxes1 = np.random.rand(N, 4).astype(np.float32) * 100
    boxes2 = np.random.rand(M, 4).astype(np.float32) * 100
    
    start = time.time()
    ious = cv_native.vectorized_iou(boxes1, boxes2)
    end = time.time()
    
    print(f"  Computed {N}x{M} IoUs in {end - start:.4f} seconds")
    print(f"  Result shape: {ious.shape}")
    assert ious.shape == (N, M)

def test_polygon_iou():
    print("\nTesting Polygon IoU...")
    # Square 1: (0,0) to (10,10)
    p1 = cv_native.PyPolygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)])
    # Square 2: (5,5) to (15,15)
    p2 = cv_native.PyPolygon([(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0), (5.0, 5.0)])
    
    # Intersection square: (5,5) to (10,10) -> area 25
    # Union area: 100 + 100 - 25 = 175
    # IoU: 25 / 175 = 0.142857
    
    iou = cv_native.polygon_iou(p1, p2)
    print(f"  Polygon Union Area: {p1.area() + p2.area() - (p1.area() * p2.area())} (Approx)") # Just debug
    print(f"  Polygon IoU: {iou:.6f} (Expected: 0.142857)")
    
    assert abs(iou - 0.142857) < 1e-4

    # Test non-overlapping
    p3 = cv_native.PyPolygon([(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0), (100.0, 100.0)])
    iou_zero = cv_native.polygon_iou(p1, p3)
    print(f"  Non-overlapping IoU: {iou_zero:.6f}")
    assert iou_zero < 1e-9

def test_spatial_index():
    print("\nTesting Spatial Index (R-Tree)...")
    polygons = []
    # Create a grid of 10x10 polygons
    for i in range(10):
        for j in range(10):
            x = i * 10
            y = j * 10
            # Square of size 8x8 at (x,y)
            p = cv_native.PyPolygon([(x, y), (x+8, y), (x+8, y+8), (x, y+8), (x, y)])
            polygons.append(p)
    
    print(f"  Building index for {len(polygons)} polygons...")
    index = cv_native.PySpatialIndex(polygons)
    
    # Query BBox: (15, 15) to (25, 25). Should intersect (10,10) and (20,20) if wide enough?
    # Our squares are 8x8.
    # (10,10) square spans x=[10,18], y=[10,18].
    # (20,20) square spans x=[20,28], y=[20,28].
    # Query box (15,15) to (25,25) overlaps:
    # - (10,10) square? max_x(18) > min_query(15) AND min_x(10) < max_query(25). Yes.
    # - (20,20) square? min_x(20) < max_query(25) AND max_x(28) > min_query(15). Yes.
    # It should find indices for (1,1) -> index 11, and (2,2) -> index 22? No, indices are flat.
    # (1,1) is index 1*10 + 1 = 11.
    # (2,2) is index 2*10 + 2 = 22.
    
    indices = index.query(15.0, 15.0, 25.0, 25.0)
    print(f"  Query (15,15)-(25,25) found {len(indices)} indices: {indices}")
    assert len(indices) > 0
    
    # Nearest neighbor to (12, 12). Should be (1,1) polygon (center approx 14,14).
    nearest_idx = index.nearest(12.0, 12.0)
    print(f"  Nearest to (12,12) index: {nearest_idx}")
    assert nearest_idx is not None

if __name__ == "__main__":
    try:
        test_iou()
        test_vectorized_iou()
        test_polygon_iou()
        test_spatial_index()
        print("\nAll geometric tests PASSED!")
    except Exception as e:
        print(f"\nTests FAILED: {e}")
        import traceback
        traceback.print_exc()
