"""
Normal Estimation Examples — cv_native
=======================================

Demonstrates every normal estimation method available through the Python bindings.
Build the extension first:
    maturin develop --release   # or: pip install maturin && maturin develop

Usage:
    python examples/normals_example.py
"""

import math
import time

try:
    import cv_native
except ImportError:
    print("Build the cv_native extension first: maturin develop --release")
    raise


# ── helpers ───────────────────────────────────────────────────────────────────

def sphere_cloud(n: int, noise: float = 0.005):
    """Generate n points on the unit sphere surface with optional noise."""
    import random
    random.seed(42)
    pts = []
    for _ in range(n):
        theta = random.uniform(0, 2 * math.pi)
        phi   = math.acos(random.uniform(-1, 1))
        r     = 1.0 + random.gauss(0, noise)
        pts.append((
            r * math.sin(phi) * math.cos(theta),
            r * math.sin(phi) * math.sin(theta),
            r * math.cos(phi),
        ))
    return pts


def flat_depth_image(width: int, height: int, depth_value: float = 1.0):
    """Flat depth image — all pixels at the same depth."""
    return [depth_value] * (width * height)


def sphere_depth_image(width: int, height: int):
    """Depth image of a sphere cap centred in the frame."""
    depth = []
    cx, cy = width / 2.0, height / 2.0
    for py in range(height):
        for px in range(width):
            dx = (px - cx) / cx   # normalised to [-1, 1]
            dy = (py - cy) / cy
            r2 = dx * dx + dy * dy
            depth.append(math.sqrt(max(1.0 - r2, 0.0)) if r2 < 1.0 else 0.0)
    return depth


def bench(label: str, fn, *args):
    """Run fn(*args), report wall-clock time and return result."""
    t0 = time.perf_counter()
    result = fn(*args)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label:<40s}  {ms:6.1f} ms  →  {len(result)} normals")
    return result


# ── point cloud methods ───────────────────────────────────────────────────────

def demo_point_cloud(n: int = 40_000, k: int = 15):
    print(f"\n=== Point cloud ({n:,} pts, k={k}) ===\n")
    pts = sphere_cloud(n)

    bench("estimate_normals_auto",             cv_native.estimate_normals_auto,            pts, k)
    bench("estimate_normals_cpu",              cv_native.estimate_normals_cpu,             pts, k)
    bench("estimate_normals_gpu",              cv_native.estimate_normals_gpu,             pts, k)
    bench("estimate_normals_hybrid",           cv_native.estimate_normals_hybrid,          pts, k)
    bench("estimate_normals_approx_cross",     cv_native.estimate_normals_approx_cross,    pts)
    bench("estimate_normals_approx_integral",  cv_native.estimate_normals_approx_integral, pts)


# ── PyPointCloud class API ────────────────────────────────────────────────────

def demo_point_cloud_class(n: int = 10_000, k: int = 15):
    print(f"\n=== PyPointCloud class API ({n:,} pts) ===\n")
    pts = sphere_cloud(n)
    cloud = cv_native.PyPointCloud(pts)

    print(f"  Cloud has {cloud.num_points()} points, normals={cloud.has_normals()}")

    # estimate in-place — default method is "auto"
    cloud.estimate_normals(k=k)
    print(f"  After estimate_normals(k={k}): normals={cloud.has_normals()}")

    normals = cloud.get_normals()   # list of (nx, ny, nz)
    # Spot-check: normals on a sphere should be roughly unit-length and radial.
    n0 = normals[0]
    length = math.sqrt(n0[0]**2 + n0[1]**2 + n0[2]**2)
    print(f"  First normal: {n0[0]:.3f}, {n0[1]:.3f}, {n0[2]:.3f}  (length={length:.4f})")

    # Flat output for numpy
    flat = cloud.get_normals_flat()  # [nx0,ny0,nz0, nx1,ny1,nz1, ...]
    print(f"  get_normals_flat() returned {len(flat)} floats")

    # Try different methods
    for method in ("cpu", "gpu", "approx_cross", "approx_integral"):
        cloud.estimate_normals(k=k, method=method)
        print(f"  estimate_normals(method='{method}'): ok, {len(cloud.get_normals())} normals")


# ── depth image normals ───────────────────────────────────────────────────────

def demo_depth_image():
    print("\n=== Depth image normals (O(n)) ===\n")

    for (w, h) in [(32, 32), (100, 100), (320, 240), (640, 480)]:
        depth = sphere_depth_image(w, h)
        # Pinhole intrinsics: fx=fy=image_width (≈ 45° FoV)
        fx = fy = float(w)
        cx, cy  = w / 2.0, h / 2.0
        bench(
            f"estimate_normals_from_depth  {w}×{h}",
            cv_native.estimate_normals_from_depth,
            depth, w, h, fx, fy, cx, cy,
        )

    # Verify: centre pixel of a sphere cap should face the viewer (|nz| > 0.8)
    w, h = 64, 64
    depth = sphere_depth_image(w, h)
    normals = cv_native.estimate_normals_from_depth(
        depth, w, h, float(w), float(h), w/2.0, h/2.0
    )
    cx_idx = (h // 2) * w + (w // 2)
    n = normals[cx_idx]
    print(f"\n  Centre pixel normal = ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})"
          f"  (|nz|={abs(n[2]):.3f}, expect > 0.8)")


# ── numpy integration example ─────────────────────────────────────────────────

def demo_numpy():
    """Show how to integrate with numpy (optional dependency)."""
    try:
        import numpy as np
    except ImportError:
        print("\n(numpy not installed — skipping numpy demo)")
        return

    print("\n=== NumPy integration ===\n")

    n = 5_000
    # Generate a flat point cloud using numpy
    rng = np.random.default_rng(42)
    xy  = rng.uniform(0, 1, (n, 2))
    z   = np.zeros((n, 1))
    pts_np = np.hstack([xy, z])   # shape (n, 3), z=0 plane

    # Convert to list-of-tuples for the API, or use PyPointCloud
    pts = [tuple(p) for p in pts_np.tolist()]
    normals_raw = cv_native.estimate_normals_auto(pts, k=10)

    # Convert result back to numpy
    normals_np = np.array(normals_raw)   # shape (n, 3)
    print(f"  Input  shape: {pts_np.shape}")
    print(f"  Output shape: {normals_np.shape}")
    print(f"  Mean |nz|: {np.abs(normals_np[:, 2]).mean():.4f}  (should be ≈ 1.0 for z=0 plane)")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("cv_native normal estimation examples")
    print("=====================================")

    demo_point_cloud(n=40_000, k=15)
    demo_point_cloud_class(n=10_000, k=15)
    demo_depth_image()
    demo_numpy()

    print("\nDone.")
