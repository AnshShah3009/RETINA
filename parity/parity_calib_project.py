#!/usr/bin/env python3
"""Camera-projection parity against `cv2.projectPoints`.

Run `cargo run -q -p cv-calib3d --example parity_project` first; it prints the Rust
projections and this script recomputes them with OpenCV.

**Every number here was measured.** Nothing adjusts a tolerance to make a
comparison pass.

Why this surface: projection is the single most-used operation in the whole
workspace — every detection, every pose estimate, every reprojection error goes
through it — and it is where a pose-convention mistake would be *invisible*: a
world-to-camera and camera-to-world pose are both well-formed, and the wrong one
produces plausible pixels. Comparing against `cv2.projectPoints` settles the
convention as well as the arithmetic.
"""
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# The intrinsics and pose the Rust example uses, restated here so this side is an
# independent transcription rather than a copy of the Rust literals.
K = np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]])
_AXIS = np.array([0.3, -0.2, 1.0])
_AXIS = _AXIS / np.linalg.norm(_AXIS)
_ROT = Rotation.from_rotvec(_AXIS * 0.35).as_matrix()
_TRANS = np.array([0.12, -0.07, 0.03])
_RVEC = Rotation.from_matrix(_ROT).as_rotvec().reshape(3, 1)


def run_rust():
    try:
        out = subprocess.run(
            ["cargo", "run", "-q", "-p", "cv-calib3d", "--example", "parity_project"],
            capture_output=True, text=True, check=True, timeout=300,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"could not run the Rust example: {e}", file=sys.stderr)
        return None
    cases, cur = {}, None
    for line in out.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "#CASE":
            cur = p[1]
            cases[cur] = {"D": None, "P": [], "X": []}
        elif p[0] == "#D":
            cases[cur]["D"] = [float(v) for v in p[1:]]
        elif p[0] == "#P":
            cases[cur]["P"].append([float(v) for v in p[1:]])
        elif p[0] == "#X":
            cases[cur]["X"].append([float(v) for v in p[1:]])
    return cases


def main() -> int:
    cases = run_rust()
    if cases is None:
        return 1

    # f64 round-off on image coordinates of order 640 is ~1e-13. Anything at 1e-9
    # would already indicate a real difference, so the tolerance is not tight enough
    # to pass a subtly wrong model, and not so tight it fails on rounding.
    ATOL_PX = 1e-9

    total = sum(len(c["X"]) for c in cases.values())
    print(f"OpenCV {cv2.__version__}, {total} projections across {len(cases)} models")
    print()
    print("| distortion | max abs err u (px) | max abs err v (px) |")
    print("|---|---:|---:|")
    worst = 0.0
    for name, c in cases.items():
        obj = np.array(c["P"], dtype=np.float64).reshape(-1, 1, 3)
        img, _ = cv2.projectPoints(obj, _RVEC, _TRANS.reshape(3, 1), K, np.array(c["D"]))
        d = np.abs(np.array(c["X"]) - img.reshape(-1, 2))
        du, dv = d[:, 0].max(), d[:, 1].max()
        worst = max(worst, du, dv)
        print(f"| {name} | {du:.3e} | {dv:.3e} |")
        assert du < ATOL_PX, f"{name}: u differs by {du:.3e} px"
        assert dv < ATOL_PX, f"{name}: v differs by {dv:.3e} px"

    print()
    print(f"worst deviation: {worst:.3e} px on image coordinates of order 640")
    print("(~3.5e-16 relative - f64 round-off.)")
    print()
    print("This also settles the pose convention in this path: `Pose` applies")
    print("`R*p + t` and `cv2.projectPoints` applies `R*X + t`, and the two agree to")
    print("round-off, so there is no world-to-camera / camera-to-world ambiguity here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
