#!/usr/bin/env python3
"""
Test Fixture Generator for cv-hal

Generates reference test fixtures using OpenCV for comparison testing.
This script should be run when updating or validating algorithm implementations.

Usage:
    python generate_fixtures.py [--download-optical-flow] [--all]

Requirements:
    pip install numpy opencv-python open3d
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import numpy as np

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available, resize fixtures will use numpy fallback")

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available, ICP fixtures will use numpy fallback")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = SCRIPT_DIR


def generate_gradient_image(width, height):
    """Generate a simple gradient image for testing."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return (xx + yy).astype(np.float32)


def generate_step_image(width, height, edge_x=None):
    """Generate a step edge image."""
    if edge_x is None:
        edge_x = width // 2
    img = np.zeros((height, width), dtype=np.float32)
    img[:, edge_x:] = 1.0
    return img


def generate_checkerboard(width, height, square_size=8):
    """Generate a checkerboard pattern."""
    img = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y, x] = 1.0
    return img


def generate_translation_pair(width, height, dx, dy):
    """Generate two images with a known translation for optical flow testing."""
    img1 = generate_gradient_image(width, height)
    img2 = np.zeros_like(img1)

    for y in range(height):
        for x in range(width):
            src_x = int(x - dx)
            src_y = int(y - dy)
            if 0 <= src_x < width and 0 <= src_y < height:
                img2[y, x] = img1[src_y, src_x]

    return img1, img2


def generate_resize_fixtures():
    """Generate reference outputs for resize operations."""
    print("Generating resize fixtures...")

    test_cases = [
        # (width, height, scale, name)
        (100, 100, 0.5, "downsample_half"),
        (100, 100, 2.0, "upscale_double"),
        (100, 100, 0.25, "downsample_quarter"),
        (99, 99, 0.5, "downsample_odd"),
        (100, 100, 3.0, "upscale_triple"),
        (100, 80, 0.5, "downsample_rect"),
    ]

    os.makedirs(os.path.join(FIXTURES_DIR, "resize"), exist_ok=True)

    for orig_w, orig_h, scale, name in test_cases:
        img = generate_gradient_image(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        np.save(os.path.join(FIXTURES_DIR, "resize", f"{name}_input.npy"), img)

        if OPENCV_AVAILABLE:
            # Bilinear interpolation
            bilinear = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            np.save(
                os.path.join(FIXTURES_DIR, "resize", f"{name}_bilinear.npy"), bilinear
            )

            # Lanczos-4 interpolation
            if scale < 1.0:
                lanczos = cv2.resize(
                    img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
                )
            else:
                # LANCZOS4 only for downsampling in OpenCV, use CUBIC for upscaling
                lanczos = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            np.save(
                os.path.join(FIXTURES_DIR, "resize", f"{name}_lanczos.npy"), lanczos
            )

            # Bicubic interpolation
            bicubic = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            np.save(
                os.path.join(FIXTURES_DIR, "resize", f"{name}_bicubic.npy"), bicubic
            )

            print(f"  Generated {name}: {orig_w}x{orig_h} -> {new_w}x{new_h}")
        else:
            # NumPy-based bilinear approximation
            x_indices = np.linspace(0, orig_w - 1, new_w)
            y_indices = np.linspace(0, orig_h - 1, new_h)
            xx, yy = np.meshgrid(x_indices, y_indices)

            x0 = xx.astype(int).clip(0, orig_w - 1)
            y0 = yy.astype(int).clip(0, orig_h - 1)
            x1 = (x0 + 1).clip(0, orig_w - 1)
            y1 = (y0 + 1).clip(0, orig_h - 1)

            fx = xx - x0.astype(float)
            fy = yy - y0.astype(float)

            bilinear = (
                (1 - fx) * (1 - fy) * img[y0, x0]
                + fx * (1 - fy) * img[y0, x1]
                + (1 - fx) * fy * img[y1, x0]
                + fx * fy * img[y1, x1]
            )

            np.save(
                os.path.join(FIXTURES_DIR, "resize", f"{name}_bilinear.npy"), bilinear
            )
            print(
                f"  Generated {name} (numpy fallback): {orig_w}x{orig_h} -> {new_w}x{new_h}"
            )


def generate_optical_flow_fixtures():
    """Generate optical flow test fixtures."""
    print("Generating optical flow fixtures...")

    os.makedirs(os.path.join(FIXTURES_DIR, "optical_flow"), exist_ok=True)

    # Translation test cases
    translations = [
        (5.0, 0.0, "right_5"),
        (0.0, 5.0, "down_5"),
        (2.5, 2.5, "diagonal_35"),
        (10.0, 0.0, "right_10"),
        (-3.0, 2.0, "left_up"),
    ]

    for dx, dy, name in translations:
        img1, img2 = generate_translation_pair(100, 100, dx, dy)
        np.save(os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_frame1.npy"), img1)
        np.save(os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_frame2.npy"), img2)

        # Expected flow field (constant translation)
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        flow[:, :, 0] = dx
        flow[:, :, 1] = dy
        np.save(os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_flow_gt.npy"), flow)

        print(f"  Generated translation test: {name} (dx={dx}, dy={dy})")

    # Add noise test case
    img1 = (
        generate_gradient_image(100, 100)
        + np.random.randn(100, 100).astype(np.float32) * 0.1
    )
    img2 = img1.copy()
    np.save(os.path.join(FIXTURES_DIR, "optical_flow", "noisy_frame1.npy"), img1)
    np.save(os.path.join(FIXTURES_DIR, "optical_flow", "noisy_frame2.npy"), img2)
    print("  Generated noisy test case")

    # Generate TVL1 reference outputs if OpenCV is available
    if OPENCV_AVAILABLE:
        print("\n  Generating TVL1 reference outputs...")

        # TVL1 test cases with larger images for better comparison
        tvl1_test_cases = [
            (256, 256, 3.0, 2.0, "tvl1_translation_3x2"),
            (256, 256, 5.0, 5.0, "tvl1_diagonal_5x5"),
            (256, 256, -2.0, 3.0, "tvl1_complex"),
        ]

        for w, h, dx, dy, name in tvl1_test_cases:
            img1 = generate_gradient_image(w, h)
            img2 = np.zeros((h, w), dtype=np.float32)

            for y in range(h):
                for x in range(w):
                    src_x = int(x - dx)
                    src_y = int(y - dy)
                    if 0 <= src_x < w and 0 <= src_y < h:
                        img2[y, x] = img1[src_y, src_x]

            np.save(
                os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_frame1.npy"), img1
            )
            np.save(
                os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_frame2.npy"), img2
            )

            # Compute TVL1 flow using OpenCV
            prev_gray = cv2.cvtColor(
                np.stack([img1] * 3, axis=-1).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
            next_gray = cv2.cvtColor(
                np.stack([img2] * 3, axis=-1).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )

            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            flow = tvl1.calc(prev_gray, next_gray, None)

            # Save as [2, H, W] format
            flow_hwc = flow.transpose(2, 0, 1)  # [2, H, W]
            np.save(
                os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_tvl1_ref.npy"),
                flow_hwc,
            )

            print(f"    Generated TVL1 ref: {name}")

        # Also generate Farneback reference for comparison
        print("\n  Generating Farnebäck reference outputs...")

        farneback_cases = [
            (256, 256, 3.0, 2.0, "farneback_translation_3x2"),
            (256, 256, 5.0, 5.0, "farneback_diagonal_5x5"),
        ]

        for w, h, dx, dy, name in farneback_cases:
            img1 = generate_gradient_image(w, h)
            img2 = np.zeros((h, w), dtype=np.float32)

            for y in range(h):
                for x in range(w):
                    src_x = int(x - dx)
                    src_y = int(y - dy)
                    if 0 <= src_x < w and 0 <= src_y < h:
                        img2[y, x] = img1[src_y, src_x]

            prev_gray = cv2.cvtColor(
                np.stack([img1] * 3, axis=-1).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
            next_gray = cv2.cvtColor(
                np.stack([img2] * 3, axis=-1).astype(np.uint8), cv2.COLOR_RGB2GRAY
            )

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
            )

            flow_hwc = flow.transpose(2, 0, 1)  # [2, H, W]
            np.save(
                os.path.join(FIXTURES_DIR, "optical_flow", f"{name}_farneback_ref.npy"),
                flow_hwc,
            )

            print(f"    Generated Farnebäck ref: {name}")


def download_sintel_optical_flow():
    """Download Sintel optical flow test data."""
    print("Downloading Sintel optical flow fixtures...")

    sintel_url = "http://files.is.tue.mpg.de/downloads/sintel-complete.zip"
    sintel_dir = os.path.join(FIXTURES_DIR, "optical_flow", "sintel")

    # Note: Sintel is a large dataset (~1.5GB)
    # For CI, we use the small fixtures above
    # This function can be used for full validation

    print("  Sintel download skipped (large dataset)")
    print("  Use small fixtures for CI, this for comprehensive validation")


def generate_icp_fixtures():
    """Generate ICP test fixtures."""
    print("Generating ICP fixtures...")

    os.makedirs(os.path.join(FIXTURES_DIR, "icp"), exist_ok=True)
    os.makedirs(os.path.join(FIXTURES_DIR, "icp", "point_clouds"), exist_ok=True)

    if OPEN3D_AVAILABLE:
        # Generate point cloud pairs with known transforms
        test_transforms = [
            (np.eye(4), "identity"),
            (
                np.array([[1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "translate_x_01",
            ),
            (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                "translate_y_01",
            ),
            (
                np.array(
                    [
                        [0.995, -0.1, 0, 0],
                        [0.1, 0.995, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                "rotate_small",
            ),
        ]

        # Create a simple point cloud (plane with some noise)
        pcd = o3d.geometry.PointCloud()
        points = np.random.randn(1000, 3).astype(np.float32)
        points[:, 2] = 0  # Flatten to plane
        pcd.points = o3d.utility.Vector3dVector(points)

        for transform, name in test_transforms:
            source = pcd
            target = pcd.clone()

            # Apply transform to source
            source.transform(transform)

            # Run ICP
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                0.1,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )

            # Save
            np.save(
                os.path.join(FIXTURES_DIR, "icp", f"{name}_source.npy"),
                np.asarray(source.points).astype(np.float32),
            )
            np.save(
                os.path.join(FIXTURES_DIR, "icp", f"{name}_target.npy"),
                np.asarray(target.points).astype(np.float32),
            )
            np.save(
                os.path.join(FIXTURES_DIR, "icp", f"{name}_expected_transform.npy"),
                result.transformation.astype(np.float32),
            )

            print(f"  Generated ICP test: {name}")
    else:
        # NumPy fallback
        source = np.random.randn(500, 3).astype(np.float32) * 10
        transform_gt = np.array(
            [[1, 0, 0, 5], [0, 1, 0, 3], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        target = source @ transform_gt[:3, :3].T + transform_gt[:3, 3]

        np.save(os.path.join(FIXTURES_DIR, "icp", "translation_source.npy"), source)
        np.save(os.path.join(FIXTURES_DIR, "icp", "translation_target.npy"), target)
        np.save(
            os.path.join(FIXTURES_DIR, "icp", "translation_expected_transform.npy"),
            transform_gt,
        )

        print("  Generated ICP test (numpy fallback): translation")


def generate_all_fixtures():
    """Generate all test fixtures."""
    print("=" * 60)
    print("Generating test fixtures for cv-hal")
    print("=" * 60)

    generate_resize_fixtures()
    generate_optical_flow_fixtures()
    generate_icp_fixtures()

    print("=" * 60)
    print("Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURES_DIR}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate test fixtures for cv-hal")
    parser.add_argument("--all", action="store_true", help="Generate all fixtures")
    parser.add_argument(
        "--resize", action="store_true", help="Generate resize fixtures"
    )
    parser.add_argument(
        "--optical-flow", action="store_true", help="Generate optical flow fixtures"
    )
    parser.add_argument("--icp", action="store_true", help="Generate ICP fixtures")
    parser.add_argument(
        "--download-sintel", action="store_true", help="Download Sintel dataset"
    )

    args = parser.parse_args()

    if not any(
        [args.all, args.resize, args.optical_flow, args.icp, args.download_sintel]
    ):
        # Default: generate all small fixtures
        args.all = True

    if args.all or args.resize:
        generate_resize_fixtures()

    if args.all or args.optical_flow:
        generate_optical_flow_fixtures()

    if args.all or args.icp:
        generate_icp_fixtures()

    if args.download_sintel:
        download_sintel_optical_flow()


if __name__ == "__main__":
    main()
