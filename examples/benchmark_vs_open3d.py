"""
Normal Estimation Benchmark: cv_native (Rust) vs Open3D (Python/C++)
=====================================================================

Measures wall-clock time for surface normal estimation at three cloud sizes.
Run after: maturin develop --release -m python/Cargo.toml

Usage:
    python examples/benchmark_vs_open3d.py
"""

import math
import time

import numpy as np
import open3d as o3d
import cv_native


# ── helpers ───────────────────────────────────────────────────────────────────

def sphere_cloud_np(n: int, seed: int = 42) -> np.ndarray:
    """Uniform sphere surface, returns (n,3) float32 array."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
    phi   = np.arccos(rng.uniform(-1, 1, n)).astype(np.float32)
    r     = (1.0 + rng.normal(0, 0.005, n)).astype(np.float32)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def measure(label: str, fn, reps: int = 3):
    """Run fn() `reps` times, return median ms."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    med = sorted(times)[len(times) // 2]
    print(f"  {label:<50s}  {med:7.1f} ms")
    return result, med


# ── Open3D methods ────────────────────────────────────────────────────────────

def open3d_estimate_normals(pts_np: np.ndarray, k: int = 15):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals)


def open3d_estimate_normals_radius(pts_np: np.ndarray, radius: float = 0.15):
    """Open3D radius search (sometimes faster for uniform clouds)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    return np.asarray(pcd.normals)


# ── cv_native methods ─────────────────────────────────────────────────────────

def rust_estimate_normals_auto(pts_np: np.ndarray, k: int = 15):
    pts = [tuple(p) for p in pts_np.tolist()]
    return np.array(cv_native.estimate_normals_auto(pts, k=k))


def rust_estimate_normals_cpu(pts_np: np.ndarray, k: int = 15):
    pts = [tuple(p) for p in pts_np.tolist()]
    return np.array(cv_native.estimate_normals_cpu(pts, k=k))


def rust_estimate_normals_gpu(pts_np: np.ndarray, k: int = 15):
    pts = [tuple(p) for p in pts_np.tolist()]
    return np.array(cv_native.estimate_normals_gpu(pts, k=k))


def rust_estimate_normals_approx(pts_np: np.ndarray):
    pts = [tuple(p) for p in pts_np.tolist()]
    return np.array(cv_native.estimate_normals_approx_cross(pts))


def rust_class_api(pts_np: np.ndarray, k: int = 15, method: str = "auto"):
    """Test the PyPointCloud class API."""
    pts = [tuple(p) for p in pts_np.tolist()]
    cloud = cv_native.PyPointCloud(pts)
    cloud.estimate_normals(k=k, method=method)
    return np.array(cloud.get_normals())


# ── NumPy-native paths (no Python list/tuple conversion) ─────────────────────

def rust_np_auto(pts_np: np.ndarray, k: int = 15):
    return cv_native.estimate_normals_np(pts_np, k=k)

def rust_np_cpu(pts_np: np.ndarray, k: int = 15):
    return cv_native.estimate_normals_cpu_np(pts_np, k=k)

def rust_np_gpu(pts_np: np.ndarray, k: int = 15):
    return cv_native.estimate_normals_gpu_np(pts_np, k=k)

def rust_np_approx(pts_np: np.ndarray):
    return cv_native.estimate_normals_approx_cross_np(pts_np)


# ── quality check ─────────────────────────────────────────────────────────────

def check_quality(normals_np: np.ndarray, pts_np: np.ndarray, label: str):
    """For a sphere, every normal should be roughly radial (|n·p_hat| > 0.7)."""
    p_hat = pts_np / (np.linalg.norm(pts_np, axis=1, keepdims=True) + 1e-9)
    dot = np.abs((normals_np * p_hat).sum(axis=1))
    good_pct = (dot > 0.7).mean() * 100
    print(f"    quality ({label}): {good_pct:.1f}% normals radial (|n·p̂| > 0.7)")


# ── main ──────────────────────────────────────────────────────────────────────

def run_benchmark(n: int, k: int = 15):
    print(f"\n{'='*65}")
    print(f" {n:,} points  (k={k})")
    print(f"{'='*65}")

    pts = sphere_cloud_np(n)

    print("\n  --- Open3D ---")
    o3d_normals, t_o3d_knn = measure(
        f"open3d  kNN  k={k}",
        lambda: open3d_estimate_normals(pts, k=k),
    )
    _, t_o3d_rad = measure(
        f"open3d  radius=0.15",
        lambda: open3d_estimate_normals_radius(pts, radius=0.15),
    )

    print("\n  --- cv_native (Rust) — list/tuple API ---")
    rust_auto,  t_rust_auto  = measure("rust  list  auto",    lambda: rust_estimate_normals_auto(pts, k=k))
    rust_cpu,   t_rust_cpu   = measure("rust  list  cpu",     lambda: rust_estimate_normals_cpu(pts, k=k))
    rust_gpu,   t_rust_gpu   = measure("rust  list  gpu",     lambda: rust_estimate_normals_gpu(pts, k=k))
    rust_approx,t_rust_approx= measure("rust  list  approx",  lambda: rust_estimate_normals_approx(pts))

    print("\n  --- cv_native (Rust) — NumPy array API (zero Python-object overhead) ---")
    rust_np_a,  t_np_auto   = measure("rust  numpy auto",     lambda: rust_np_auto(pts, k=k))
    rust_np_c,  t_np_cpu    = measure("rust  numpy cpu",      lambda: rust_np_cpu(pts, k=k))
    rust_np_g,  t_np_gpu    = measure("rust  numpy gpu",      lambda: rust_np_gpu(pts, k=k))
    rust_np_x,  t_np_approx = measure("rust  numpy approx",   lambda: rust_np_approx(pts))

    print("\n  --- Speedup vs Open3D kNN ---")
    def speedup(t_rust, label):
        s = t_o3d_knn / t_rust
        flag = "faster" if s > 1 else "slower"
        print(f"  {label:<44s}  {s:5.2f}× {flag}")

    speedup(t_np_auto,   "numpy auto   vs open3d kNN")
    speedup(t_np_cpu,    "numpy cpu    vs open3d kNN")
    speedup(t_np_gpu,    "numpy gpu    vs open3d kNN")
    speedup(t_np_approx, "numpy approx vs open3d kNN")
    speedup(t_rust_auto, "list  auto   vs open3d kNN  (includes py conversion)")

    print("\n  --- Quality check (sphere normals should be radial) ---")
    check_quality(o3d_normals, pts, "open3d kNN")
    check_quality(rust_np_a,   pts, "rust numpy auto")
    check_quality(rust_np_c,   pts, "rust numpy cpu")
    check_quality(rust_np_x,   pts, "rust numpy approx_cross")


def run_depth_benchmark():
    """Compare depth image normals: cv_native O(n) vs Open3D from point cloud."""
    print(f"\n{'='*65}")
    print(f" Depth image normals — RGBD camera path")
    print(f"{'='*65}")

    for (w, h) in [(100, 100), (320, 240), (640, 480)]:
        # Synthetic sphere-cap depth image
        cx, cy = w / 2.0, h / 2.0
        xs = (np.arange(w) - cx) / cx
        ys = (np.arange(h) - cy).reshape(-1, 1) / cy
        r2 = xs**2 + ys**2
        depth = np.where(r2 < 1.0, np.sqrt(np.maximum(1.0 - r2, 0.0)), 0.0).astype(np.float32).ravel()
        fx = fy = float(w)

        print(f"\n  {w}×{h}  ({w*h:,} pixels)")

        # cv_native: O(n) structured depth path
        _, t_rust = measure(
            f"rust  estimate_normals_from_depth",
            lambda: cv_native.estimate_normals_from_depth(
                depth.tolist(), w, h, fx, fy, cx, cy
            ),
        )

        # Open3D equivalent: back-project depth → point cloud → kNN normals
        # (This is what you'd have to do in Open3D — there's no structured depth path)
        def o3d_from_depth():
            # Create RGBD image and unproject
            depth_o3d = o3d.geometry.Image((depth.reshape(h, w) * 1000).astype(np.uint16))
            color_o3d = o3d.geometry.Image(np.ones((h, w, 3), dtype=np.uint8) * 128)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
            return pcd

        _, t_o3d = measure(f"open3d  RGBD → pointcloud → kNN normals", o3d_from_depth)

        speedup = t_o3d / t_rust
        print(f"  → rust depth path is {speedup:.1f}× faster than open3d RGBD pipeline")


if __name__ == "__main__":
    print("cv_native vs Open3D — Normal Estimation Benchmark")
    print("==================================================")
    print(f"cv_native version: {getattr(cv_native, '__version__', 'dev')}")
    print(f"open3d   version: {o3d.__version__}")
    print(f"numpy    version: {np.__version__}")

    for n in [5_000, 20_000, 40_000]:
        run_benchmark(n, k=15)

    run_depth_benchmark()

    print("\nDone.")
