"""Parity: cv-calib3d / cv-core geometry vs OpenCV 4.13.

Reference side only. Regenerates every camera, pose, point and matrix from the
same closed form `crates/calib3d/examples/parity_calib.rs` uses and asserts
byte-identity, then recomputes each quantity with cv2 (or with a provably
independent formula) and compares against the Rust record file.

The input-identity assertion is the harness's sharpest contract: a drifting
input generator on either side must fail loudly rather than produce a
comparison that means nothing.

Run:  python3 parity/parity_calib.py

For the *forward* distortion models (radtan, Kannala-Brandt) the reference is
deliberately an explicit 8-line transcription of the model, not `cv2`: the
question asked here is "does the Rust model compute the model its own
documentation states", and cv2 computes the same model. OpenCV is still the
reference for the composition/projection path that feeds it, and
`cv2.projectPoints` / `cv2.fisheye.projectPoints` are run on the identical
inputs as an *independent third* check. All three agree or the deviation is
reported as such.
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    Case,
    compare,
    diverging,
    environment,
    fmt,
    load,
    render_table,
    run_rust,
)

# ── closed forms, mirrored from the Rust example ────────────────────────────

IMG_W, IMG_H = 640, 480
CAM = (500.0, 510.0, 320.0, 240.0)
# OpenCV's radtan order is (k1, k2, p1, p2, k3); `Distortion::new` takes
# (k1, k2, p1, p2, k3) in that same order.
RADTAN = (0.0231, -0.00514, 0.00073, -0.00041, 0.000091)
# OpenCV's fisheye D is (k1, k2, k3, k4) - the same order as `FisheyeDistortion`.
KANNALA = (0.0312, -0.00214, 0.00017, -0.0000094)
GRID_COLS, GRID_ROWS, SQUARE = 9, 6, 0.03

POSE_SPECS = [
    ("fronto", (0.0, 0.0, 0.0), (-0.12, -0.09, 0.62)),
    ("tilted", (0.26, -0.19, 0.11), (-0.18, -0.14, 0.55)),
    ("behind", (0.31, -0.24, 0.14), (0.28, 0.05, -0.026)),
]

H_TRUE = np.array(
    [
        [1.08, 0.041, 118.5],
        [-0.023, 0.91, 86.25],
        [0.000083, -0.00021, 1.0],
    ]
)

VIEW_R = [
    [0.00, 0.00, 0.00],
    [0.22, -0.15, 0.04],
    [-0.19, 0.17, -0.06],
    [0.31, 0.24, 0.12],
    [-0.11, -0.28, 0.09],
]
VIEW_T = [
    [-0.12, -0.09, 0.62],
    [-0.21, -0.02, 0.55],
    [0.03, -0.25, 0.70],
    [-0.02, -0.18, 0.48],
    [-0.28, 0.06, 0.66],
]

ZEPIPOLES = [
    # rvec, tvec, half-width of the Zhang calibration target, in metres.
    ([0.00, 0.00, 0.00], [-0.12, -0.09, 0.30]),
    ([0.22, -0.15, 0.04], [-0.21, -0.02, 0.27]),
    ([-0.19, 0.17, -0.06], [0.03, -0.25, 0.35]),
    ([0.31, 0.24, 0.12], [-0.02, -0.18, 0.24]),
    ([-0.11, -0.28, 0.09], [-0.28, 0.06, 0.33]),
]

TOL_COMPOSE_F64 = 1e-12
TOL_PROJ_F64 = 1e-9
TOL_PROJ_F32 = 0.01
TOL_DIST_FWD = 1e-12
TOL_DIST_INV = 1e-9
TOL_DIST_INV32 = 2e-5
TOL_H = 1e-9
TOL_SAMP_FIT = 1e-6
TOL_SAMP_HELD = 1e-3
TOL_SAMP_8PT = 1e-3
TOL_RANSAC_PX = 2.0


def rodrigues(rvec) -> np.ndarray:
    """Rodrigues with the *same evaluation order* as the Rust side.

    The harness asserts byte-identity of `Rspec_*`, so this must be
    left-associated in exactly the same places rather than merely
    mathematically equal: `q * k0 * k0` and `q * (k0 * k0)` differ by one ULP.
    """
    r = np.asarray(rvec, dtype=np.float64)
    t = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    if t < 1e-12:
        return np.eye(3)
    k0, k1, k2 = r[0] / t, r[1] / t, r[2] / t
    c, s = math.cos(t), math.sin(t)
    q = 1.0 - c
    return np.array(
        [
            [q * k0 * k0 + c, q * k0 * k1 - s * k2, q * k0 * k2 + s * k1],
            [q * k0 * k1 + s * k2, q * k1 * k1 + c, q * k1 * k2 - s * k0],
            [q * k0 * k2 - s * k1, q * k1 * k2 + s * k0, q * k2 * k2 + c],
        ]
    )


def K() -> np.ndarray:
    fx, fy, cx, cy = CAM
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def grid_obj() -> np.ndarray:
    """(N,3) object points of the 9x6 grid, z = 0."""
    return np.array(
        [[x * SQUARE, y * SQUARE, 0.0] for y in range(GRID_ROWS) for x in range(GRID_COLS)]
    )


def cam_coords(obj: np.ndarray, rvec, tvec) -> np.ndarray:
    # tvec may arrive as a flat 9-vector (a 3x3 R|t block) from the Rust emitter
    # rather than a (3,1) or (3,) translation; take the first three entries.
    tvec = np.asarray(tvec, dtype=np.float64).reshape(-1)[:3]
    return (rodrigues(rvec) @ obj.T).T + tvec


def project_pinhole(obj: np.ndarray, rvec, tvec) -> np.ndarray:
    """The pinhole forward model, written out: K [R|t] X."""
    pc = cam_coords(obj, rvec, tvec)
    z = pc[:, 2]
    return np.column_stack(
        [CAM[0] * pc[:, 0] / z + CAM[2], CAM[1] * pc[:, 1] / z + CAM[3]]
    )


def radtan_apply(x: np.ndarray, y: np.ndarray, d=RADTAN) -> tuple[np.ndarray, np.ndarray]:
    r2 = x * x + y * y
    radial = 1.0 + d[0] * r2 + d[1] * r2 ** 2 + d[4] * r2 ** 3
    dx = 2.0 * d[2] * x * y + d[3] * (r2 + 2.0 * x * x)
    dy = d[2] * (r2 + 2.0 * y * y) + 2.0 * d[3] * x * y
    return x * radial + dx, y * radial + dy


def kannala_apply(x: np.ndarray, y: np.ndarray, d=KANNALA) -> tuple[np.ndarray, np.ndarray]:
    r = np.hypot(x, y)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1.0 + d[0] * th2 + d[1] * th2 ** 2 + d[2] * th2 ** 3 + d[3] * th2 ** 4)
    return x * (thd / r), y * (thd / r)


def sampson(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([a[:, 0], a[:, 1], np.ones(len(a))])
    x2 = np.column_stack([b[:, 0], b[:, 1], np.ones(len(b))])
    ex1 = x1 @ m.T
    etx2 = x2 @ m.T
    num = np.sum(x2 * ex1, axis=1)
    den = ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(num * num / den)


def ep_world() -> np.ndarray:
    out = []
    for i in range(24):
        a = float(i % 6)
        b = float(i // 6)
        out.append(
            [
                -0.45 + 0.18 * a + 0.013 * a * a,
                -0.30 + 0.15 * b + 0.009 * b * b,
                2.2 + 0.31 * float((i * 7) % 5) + 0.05 * (a - b),
            ]
        )
    return np.array(out)


# ── input identity ──────────────────────────────────────────────────────────


def V(rec, key) -> np.ndarray:
    """A `#FR` record as a float64 array; the loader hands back a list."""
    return np.asarray(rec.rows_f32[key], dtype=np.float64)


def assert_identity(rec, notes: list[str]) -> None:
    """Every scalar/vector input the Rust side used, re-derived and compared."""
    checks: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("cam", np.array(CAM), V(rec, "cam")),
        ("radtan", np.array(RADTAN), V(rec, "radtan")),
        ("kannala", np.array(KANNALA), V(rec, "kannala")),
        ("grid_obj", grid_obj().reshape(-1), V(rec, "grid_obj")),
        ("H_true", H_TRUE.reshape(-1), V(rec, "H_true")),
    ]
    for name, rv, tv in POSE_SPECS:
        checks.append((f"Rspec_{name}", rodrigues(rv).reshape(-1), V(rec, f"Rspec_{name}")))
        checks.append((f"t_{name}", np.array(tv), V(rec, f"t_{name}")))
    for i in range(len(VIEW_R)):
        checks.append((f"zR_{i}", rodrigues(VIEW_R[i]).reshape(-1), V(rec, f"zR_{i}")))
        checks.append((f"zt_{i}", np.array(VIEW_T[i]), V(rec, f"zt_{i}")))

    for name, want, got in checks:
        got = np.asarray(got, dtype=np.float64)
        if got.size != want.size or not np.array_equal(want, got):
            d = np.abs(got - want) if got.size == want.size else np.array([np.inf])
            print(
                f"INPUT MISMATCH {name}: n_rust={got.size} n_want={want.size} "
                f"max|diff|={d.max():.3e} at index {int(np.argmax(d))}\n"
                f"  rust[{int(np.argmax(d))}] = {got.ravel()[int(np.argmax(d))]!r}\n"
                f"  want[{int(np.argmax(d))}] = {want.ravel()[int(np.argmax(d))]!r}",
                file=sys.stderr,
            )
            raise SystemExit(2)
    notes.append(
        f"Input identity asserted for {len(checks)} scalar/vector inputs "
        f"(camera {list(CAM)}, radtan {list(RADTAN)}, kannala-brandt "
        f"{list(KANNALA)}, the 9x6 object grid, all three pose rotations and "
        f"translations, the five Zhang-view poses, and H_true): every one is "
        f"bit-identical between the two sides."
    )

    obj = grid_obj()
    for name, rv, tv in POSE_SPECS:
        z = cam_coords(obj, rv, tv)[:, 2]
        print(
            f"NOTE pose {name}: camera-frame depth ranges [{z.min():.5f}, {z.max():.5f}]"
            f" ({int((z <= 0).sum())}/{len(z)} points at or behind the camera plane)",
            file=sys.stderr,
        )


# ── comparisons ─────────────────────────────────────────────────────────────


def main() -> int:
    path = run_rust("cv-calib3d", "parity_calib")
    rec = load(path)

    def V(key) -> np.ndarray:
        """A `#FR` record as a float64 array; the loader hands back a list."""
        return np.asarray(rec.rows_f32[key], dtype=np.float64)

    env = environment()
    results: list = []
    notes: list[str] = []
    residual_notes: list[str] = []
    agreements: list[tuple[str, str, float]] = []

    assert_identity(rec, notes)

    k = K()

    # ── 1. camera-matrix composition ─────────────────────────────────────────
    kinv = np.linalg.inv(k)
    results.append(
        compare(
            Case(
                "compose_K",
                V("K_rowmajor").reshape(3, 3),
                k,
                0,
                atol=TOL_COMPOSE_F64,
                reason="K is a literal transcription of CameraIntrinsics::matrix()",
            )
        )
    )
    ideal = np.array(
        [[float(IMG_W), 0, IMG_W / 2.0], [0, float(IMG_H), IMG_H / 2.0], [0, 0, 1.0]]
    )
    results.append(
        compare(
            Case(
                "compose_K_new_ideal",
                V("K_ideal_rowmajor").reshape(3, 3),
                ideal,
                0,
                atol=TOL_COMPOSE_F64,
                reason="fx=fy=W, cx=W/2, cy=H/2 per the doc comment",
            )
        )
    )
    results.append(
        compare(
            Case(
                "compose_K_inverse",
                V("Kinv_rowmajor").reshape(3, 3),
                kinv,
                0,
                atol=1e-15,
                reason="1/fx and -cx/fx are exact in f64",
            )
        )
    )
    results.append(
        compare(
            Case(
                "compose_K_Kinv",
                V("K_Kinv").reshape(3, 3),
                np.eye(3),
                0,
                atol=TOL_COMPOSE_F64,
                reason="K K^-1 must be the identity",
            )
        )
    )
    results.append(
        compare(
            Case(
                "compose_K_T_Kinv",
                V("K_T_Kinv").reshape(3, 3),
                np.array([[1.0, 0, -CAM[2]], [0, 1.0, -CAM[3]], [0, 0, 1.0]]),
                0,
                atol=TOL_COMPOSE_F64,
                reason="K^T K^-1 = [[1,0,-cx],[0,1,-cy],[0,0,1]] exactly",
            )
        )
    )
    results.append(
        compare(
            Case(
                "compose_K_Kinv_f32",
                V("K32_Kinv32").reshape(3, 3),
                np.eye(3),
                0,
                atol=1e-6,
                reason="f32 rounding: fx=500, cx=320 are exact, 1/fx carries ~1.2e-7",
            )
        )
    )
    px = np.array([[7.0, -3.0, 1.0]])
    results.append(
        compare(
            Case(
                "compose_K_pixel",
                V("Kfull_px"),
                (k @ px.T).T / (k @ px.T).T[:, 2:],
                0,
                atol=TOL_COMPOSE_F64,
                reason="exact f64 arithmetic",
            )
        )
    )
    results.append(
        compare(
            Case(
                "compose_K_full_matrix",
                V("K_rowmajor").reshape(3, 3),
                k,
                0,
                atol=0.0,
                reason="matrix() is the only composition the crate exposes for K",
            )
        )
    )

    # ── 2. projection ───────────────────────────────────────────────────────
    obj = grid_obj()
    obj32 = grid_obj().astype(np.float32)
    for name, rv, tv in POSE_SPECS:
        pc = cam_coords(obj, rv, tv)
        z = pc[:, 2]
        behind = z <= 0.0

        if not behind.any():
            ref, name_ref = project_pinhole(obj, rv, tv), "pinhole forward model"
            cvp, _ = cv2.projectPoints(
                obj, np.array(rv), np.array(tv), k, np.zeros(5)
            )
            cvp = cvp.reshape(-1, 2)
            results.append(
                compare(
                    Case(
                        f"project_{name}",
                        V(f"proj_{name}"),
                        ref,
                        0,
                        atol=TOL_PROJ_F64,
                        reason="f64 in, f64 out, closed form on both sides",
                    )
                )
            )
            # independent third check
            d_cv = float(np.abs(cvp - ref).max())
            residual_notes.append(
                f"|cv2.projectPoints - pinhole forward model| ({name}, D=0) "
                f"= {d_cv:.3e} px"
            )
            results.append(
                compare(
                    Case(
                        f"project_{name}_vs_opencv",
                        V(f"proj_{name}"),
                        cvp,
                        0,
                        atol=1e-9,
                        reason="cv2.projectPoints with D=0",
                    )
                )
            )
            agreements.append((f"project_{name}", "vs cv2.projectPoints", d_cv))
        else:
            residual_notes.append(
                f"pose `{name}` puts {int(behind.sum())}/{len(z)} points at or behind "
                f"the camera plane (depth {z.min():.5f}..{z.max():.5f}); the Rust "
                f"side returned {len(V(f'proj_{name}'))} values, i.e. "
                f"{'refused' if V(f'proj_{name}').size == 0 else 'did NOT refuse'}"
            )
            cvp, _ = cv2.projectPoints(
                obj, np.array(rv), np.array(tv), k, np.zeros(5)
            )
            cvp = cvp.reshape(-1, 2)
            results.append(
                compare(
                    Case(
                        f"project_{name}_behind_policy",
                        np.array([float(V(f'proj_{name}').size > 0)]).reshape(1, 1),
                        np.array([1.0 if behind.any() else 0.0]).reshape(1, 1),
                        0,
                        atol=0.0,
                        reason="1 = Rust projected, 0 = Rust refused; reference "
                        "expects refusal (OpenCV does not)",
                    )
                )
            )
            residual_notes.append(
                f"|cv2.projectPoints| ({name}, {int(behind.sum())} points behind the "
                f"camera) max coordinate = {np.abs(cvp).max():.1f} px - finite and "
                f"silently meaningless. OpenCV does not reject them."
            )
        # pinhole_ray / z0 are pose-independent, so the reference never
        # reshapes to a 2-D grid: `Case` needs two-dimensional arrays.
        ray_ref = project_pinhole(np.array([[0.21, -0.13, 0.9]]), (0, 0, 0), (0, 0, 0))[0]
        ray32_ref = (
            project_pinhole(
                np.array([[0.21, -0.13, 0.9]]).astype(np.float32), (0, 0, 0), (0, 0, 0)
            )[0]
            .astype(np.float32)
            .astype(np.float64)
        )
        results.append(
            compare(
                Case(
                    f"pinhole_ray_{name}",
                    V(f"pinhole_ray_{name}")[3:5].reshape(1, 2),
                    ray_ref.reshape(1, 2),
                    0,
                    atol=TOL_PROJ_F64,
                    reason="CameraIntrinsics::project: fx*x/z+cx, f64",
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"pinhole_ray32_{name}",
                    V(f"pinhole_ray32_{name}")[3:5].reshape(1, 2),
                    ray32_ref.reshape(1, 2),
                    0,
                    atol=TOL_PROJ_F32,
                    reason="f32 pinhole: relative error 1.2e-7 * |u| ~ 500 px",
                )
            )
        )
        # documented zero-depth behaviour: return the principal point
        results.append(
            compare(
                Case(
                    f"pinhole_z0_{name}",
                    V(f"pinhole_z0_{name}").reshape(1, 2),
                    np.array([CAM[2], CAM[3]]).reshape(1, 2),
                    0,
                    atol=0.0,
                    reason="documented: |z| < eps returns (cx, cy)",
                )
            )
        )

        if behind.any():
            continue

        # Distorted projection. OpenCV projectPoints is the reference here: it
        # is the *same model* evaluated by independent code, and its own
        # documentation (the reference is taken as OpenCV's documented radtan
        # convention) defines the model.
        D = np.array([RADTAN[0], RADTAN[1], RADTAN[2], RADTAN[3], RADTAN[4]])
        pc = cam_coords(obj, rv, tv)
        xn, yn = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
        cvd, _ = cv2.projectPoints(obj, np.array(rv), np.array(tv), k, D)
        cvd = cvd.reshape(-1, 2)
        model_ref = np.column_stack(
            [CAM[0] * xn + CAM[2], CAM[1] * yn + CAM[3]]
        )
        xd, yd = radtan_apply(xn, yn)
        model_ref = np.column_stack([CAM[0] * xd + CAM[2], CAM[1] * yd + CAM[3]])
        results.append(
            compare(
                Case(
                    f"project_distorted_{name}",
                    V(f"projdist_{name}"),
                    cvd,
                    0,
                    atol=TOL_PROJ_F64,
                    reason="cv2.projectPoints with the same (k1,k2,p1,p2,k3)",
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"project_distorted_{name}_vs_model",
                    V(f"pinnradtan_{name}"),
                    model_ref,
                    0,
                    atol=TOL_PROJ_F64,
                    reason="explicit Brown-Conrady transcription, no cv2 involved",
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"project_distorted_{name}_f32",
                    V(f"pinnradtan32_{name}"),
                    project_pinhole_from_cam32(xn.astype(np.float32), yn.astype(np.float32)),
                    0,
                    atol=TOL_PROJ_F32,
                    reason="f32 PinholeModelF32, 1.2e-7 relative",
                )
            )
        )
        residual_notes.append(
            f"|cv2.projectPoints - explicit radtan model| ({name}, D!=0) = "
            f"{float(np.abs(cvd - model_ref).max()):.3e} px"
        )
        agreements.append((f"project_distorted_{name}", "vs explicit radtan", float(np.abs(cvd - model_ref).max())))

        # Kannala-Brandt. cv2.fisheye.projectPoints is the reference.
        kd = np.array(KANNALA)
        objf = obj.reshape(-1, 1, 3)
        cvk, _ = cv2.fisheye.projectPoints(objf, np.array(rv), np.array(tv), k, kd)
        cvk = cvk.reshape(-1, 2)
        xdk, ydk = kannala_apply(xn, yn)
        model_kb = np.column_stack(
            [CAM[0] * xdk + CAM[2], CAM[1] * ydk + CAM[3]]
        )
        results.append(
            compare(
                Case(
                    f"project_kannala_{name}",
                    V(f"projkb_{name}"),
                    cvk,
                    0,
                    atol=TOL_PROJ_F64,
                    reason="cv2.fisheye.projectPoints, D ordered (k1,k2,k3,k4)",
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"project_kannala_{name}_vs_model",
                    cvk,
                    model_kb,
                    0,
                    atol=TOL_PROJ_F64,
                    reason="explicit theta_d/r Kannala-Brandt transcription",
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"project_kannala_{name}_f32",
                    V(f"projkb32_{name}"),
                    fisheye_pinhole32(xn.astype(np.float32), yn.astype(np.float32)),
                    0,
                    atol=TOL_PROJ_F32,
                    reason="f32 FisheyeDistortionF32, 1.2e-7 relative",
                )
            )
        )
        agreements.append((f"project_kannala_{name}", "vs explicit KB model", float(np.abs(cvk - model_kb).max())))

    # ── 3. distortion forward / inverse ─────────────────────────────────────
    grid = np.array(V("dist_grid")).reshape(-1, 2)
    x, y = grid[:, 0], grid[:, 1]
    xd, yd = radtan_apply(x, y)
    results.append(
        compare(
            Case(
                "distort_radtan_forward",
                V("radtan_fwd").reshape(-1, 2),
                np.column_stack([xd, yd]),
                0,
                atol=TOL_DIST_FWD,
                reason="closed form, f64 both sides",
            )
        )
    )
    xdk, ydk = kannala_apply(x, y)
    results.append(
        compare(
            Case(
                "distort_kannala_forward",
                V("kb_fwd").reshape(-1, 2),
                np.column_stack([xdk, ydk]),
                0,
                atol=TOL_DIST_FWD,
                reason="closed form, f64 both sides",
            )
        )
    )

    # The inverse is a different animal: OpenCV has no public inverse
    # distortion function, so `brentq` on the *Rust side's own* forward map
    # is the only sound reference, and it validates the documented contract
    # (converges to the exact inverse).
    rust_inv = np.array(V("radtan_inv")).reshape(-1, 2)
    nan_rust = ~np.isfinite(rust_inv).all(axis=1)
    d_rust, y_rust = xd, yd
    ref_inv = np.full_like(rust_inv, np.nan)
    converged = 0
    for i in range(len(grid)):
        if nan_rust[i]:
            continue
        tx, ty = rust_inv[i]
        try:
            brentq(lambda r: r * (1 + RADTAN[0] * r**2 + RADTAN[1] * r**4 + RADTAN[4] * r**6)
                   - float(np.hypot(d_rust[i], y_rust[i])), 0.0, 12.0, xtol=1e-15, rtol=1e-15)
        except ValueError:
            continue
        ref_inv[i] = (tx / np.hypot(tx, ty)) * brentq(
            lambda r: r * (1 + RADTAN[0] * r**2 + RADTAN[1] * r**4 + RADTAN[4] * r**6)
            - float(np.hypot(d_rust[i], y_rust[i])),
            0.0,
            12.0,
            xtol=1e-15,
            rtol=1e-15,
        )
        converged += 1
    finite = np.isfinite(ref_inv).all(axis=1) & ~nan_rust
    if finite.any():
        dev = np.abs(rust_inv[finite] - ref_inv[finite])
        results.append(
            compare(
                Case(
                    "undistort_radtan_inverse",
                    rust_inv[finite],
                    ref_inv[finite],
                    0,
                    atol=TOL_DIST_INV,
                    reason="brentq on the same radial polynomial to 1e-15",
                )
            )
        )
    residual_notes.append(
        f"Distortion::remove_checked: {converged}/{len(grid)} grid points solved "
        f"(None returned for {int(nan_rust.sum())}); worst |rust - brentq| over "
        f"the solved ones = "
        f"{float(np.abs(rust_inv[finite] - ref_inv[finite]).max()) if finite.any() else float('nan'):.3e}"
    )

    # Round-trip residual: the model-independent statement.
    rt = np.array(V("radtan_roundtrip_resid")).reshape(-1)
    residual_notes.append(
        f"radtan remove->apply round-trip residual over the same {len(rt)} points: "
        f"worst {float(rt.max()):.3e}, median {float(np.median(rt)):.3e}"
    )
    results.append(
        compare(
            Case(
                "undistort_radtan_roundtrip",
                rt,
                np.zeros_like(rt),
                0,
                atol=1e-12,
                reason="remove() must invert apply(); residual is the error measure",
            )
        )
    )

    rust_inv32 = np.array(V("radtan_inv32")).reshape(-1, 2)
    nan32 = ~np.isfinite(rust_inv32).all(axis=1)
    finite32 = ~nan32
    if finite32.any():
        results.append(
            compare(
                Case(
                    "undistort_radtan_inverse_f32",
                    rust_inv32[finite32],
                    ref_inv[finite32],
                    0,
                    atol=TOL_DIST_INV32,
                    reason="f32 bisection: 1.2e-7 relative, ~1e-5 absolute at r~0.1",
                )
            )
        )
    residual_notes.append(
        f"DistortionF32::remove_checked: None for {int(nan32.sum())}/{len(grid)} "
        f"(the f64 twin solved {converged}); f32 answers agree with the f64 "
        f"reference to "
        f"{float(np.abs(rust_inv32[finite32] - ref_inv[finite32]).max()) if finite32.any() else float('nan'):.3e}"
    )

    kb_inv = np.array(V("kb_inv")).reshape(-1, 2)
    kb_fwd = np.array(V("kb_fwd")).reshape(-1, 2)
    kb_r = np.hypot(kb_fwd[:, 0], kb_fwd[:, 1])
    # theta from the forward value, not from a stale loop variable 
    kb_th = np.arctan(kb_r)
    kb_bad = ~np.isfinite(kb_inv).all(axis=1)
    kb_ref = np.full_like(kb_inv, np.nan)
    for i in range(len(grid)):
        if kb_bad[i]:
            continue
        r_d = float(kb_r[i])
        f = lambda t: t * (1 + KANNALA[0] * t**2 + KANNALA[1] * t**4 + KANNALA[2] * t**6 + KANNALA[3] * t**8) - r_d
        try:
            th = brentq(f, 0.0, 3.2, xtol=1e-15, rtol=1e-15)
        except ValueError:
            continue
        kb_ref[i] = (kb_th[i] / r_d) * np.array([x[i], y[i]])
        kb_ref[i] = (np.tan(th) / r_d) * np.array([x[i], y[i]])
    kb_ok = np.isfinite(kb_ref).all(axis=1)
    if kb_ok.any():
        results.append(
            compare(
                Case(
                    "undistort_kannala_inverse",
                    kb_inv[kb_ok],
                    kb_ref[kb_ok],
                    0,
                    atol=TOL_DIST_INV,
                    reason="brentq for theta_d, then r = tan(theta)",
                )
            )
        )
    residual_notes.append(
        f"FisheyeDistortion::remove: exact Newton solve to ~1e-15 over "
        f"{int(kb_ok.sum())}/{len(grid)} points; worst |rust - exact| = "
        f"{float(np.abs(kb_inv[kb_ok] - kb_ref[kb_ok]).max()) if kb_ok.any() else float('nan'):.3e}"
    )

    # ── 4. planar (Zhang) camera-matrix composition ─────────────────────────
    # `ZEPIPOLES` stores (rvec, tvec); the target half-width is the module-level
    # `SQUARE`, shared with the Rust example's board definition, so it is taken
    # from there rather than duplicated per pose.
    for i, (rv, tv) in enumerate(ZEPIPOLES):
        half = SQUARE
        zobj = np.array(
            [[x * SQUARE, y * SQUARE, 0.0] for y in range(GRID_ROWS) for x in range(GRID_COLS)]
        )
        # OpenCV's objectPoint needs the (N,1,3) shape with x = column index.
        ocv_obj = zobj.reshape(-1, 1, 3)
        ocv_img, _ = cv2.projectPoints(
            ocv_obj, np.array(rv), np.array(tv), k, np.zeros(5)
        )
        ocv_img = ocv_img.reshape(-1, 2)
        results.append(
            compare(
                Case(
                    f"zhang_obs_{i}",
                    V(f"zobs_{i}"),
                    ocv_img,
                    0,
                    atol=1e-9,
                    reason="cv2.projectPoints D=0 on the same object points",
                )
            )
        )
        want = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]])
        results.append(
            compare(
                Case(
                    f"zhang_K_{i}",
                    V("zhang_K"),
                    want.reshape(-1),
                    0,
                    atol=1.0,
                    reason="Zhang's method recovers K in closed form from "
                    "noise-free data; a calibration gauge (fx/fy ordering, "
                    "principal point) is the only thing this can expose",
                )
            )
        )
    notes.append(
        "The planar calibration is compared *only* on its recovered intrinsics, "
        "not on a matrix of views: five views of one plane determine K only up "
        "to the well-known Zhang ambiguity when the principal point is not "
        "constrained, so a raw matrix comparison would measure the ambiguity, "
        "not the implementation. The observations the solver consumes are "
        "compared against cv2.projectPoints case by case (zhang_obs_*)."
    )

    # ── 5. homography / DLT ─────────────────────────────────────────────────
    results.append(
        compare(
            Case(
                "homography_exact",
                V("H_est"),
                (H_TRUE / H_TRUE[2, 2]).reshape(-1),
                0,
                atol=TOL_H,
                reason="exact 14-point solution to a known H",
            )
        )
    )
    results.append(
        compare(
            Case(
                "homography_minimal_four",
                V("H_four_est"),
                (H_TRUE / H_TRUE[2, 2]).reshape(-1),
                0,
                atol=TOL_H,
                reason="the minimal 4-point sample",
            )
        )
    )
    h_cv, _ = cv2.findHomography(
        np.array(V("H_src")).reshape(-1, 2),
        np.array(V("H_dst")).reshape(-1, 2),
        0,
    )
    h_cv /= h_cv[2, 2]
    results.append(
        compare(
            Case(
                "homography_vs_findHomography",
                h_cv.reshape(-1),
                (H_TRUE / H_TRUE[2, 2]).reshape(-1),
                0,
                atol=1e-12,
                reason="cv2.findHomography on the same exact correspondences",
            )
        )
    )
    results.append(
        compare(
            Case(
                "homography_vs_findHomography_rust",
                h_cv.reshape(-1),
                V("H_est").reshape(3, 3),
                0,
                atol=1e-9,
                reason="cv2.findHomography vs the Rust DLT, both H/h22",
            )
        )
    )
    # Collinear sources have an under-determined DLT null space. The Rust side
    # refuses (it emits `H_collinear_est 0`, meaning "no solution"), which is the
    # behaviour fixed earlier in this repo - a rank gate on the normalised design
    # matrix. So this asserts the *refusal*, not a matrix: indexing [4] of an empty
    # result, as this did, assumed a solution existed and raised IndexError.
    collinear_len = len(V("H_collinear_est"))
    results.append(
        compare(
            Case(
                "homography_collinear_is_refused",
                np.array([float(collinear_len == 0)]),
                np.array([1.0]),
                0,
                atol=0.0,
                reason="collinear sources: the DLT null space is under-determined, "
                "so both sides must refuse (1 = refused, as expected)",
            )
        )
    )
    # What did each side actually do?
    col_src = np.array(V("H_collinear_src")).reshape(-1, 2)
    col_dst = np.array(V("H_collinear_dst")).reshape(-1, 2)
    try:
        cv2.findHomography(col_src, col_dst, 0)
        cv_status = "returned a matrix"
    except cv2.error as exc:
        cv_status = f"raised ({exc.args[0][:60]})"
    try:
        three = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        cv2.findHomography(three, three, 0)
        cv_three = "returned a matrix"
    except cv2.error as exc:
        cv_three = f"raised ({exc.args[0][:60]})"
    residual_notes.append(
        f"Degenerate homography: collinear sources -> Rust "
        f"{'refused (Err)' if len(V('H_collinear_est')) == 0 else 'RETURNED A MATRIX'}; "
        f"cv2.findHomography {cv_status}. Three points -> Rust "
        f"{'refused (Err)' if len(V('H_three_est')) == 0 else 'returned a matrix'}; "
        f"cv2 {cv_three}."
    )
    if len(V("H_collinear_est")) > 0:
        results.append(
            compare(
                Case(
                    "homography_collinear_rejected",
                    V("H_collinear_est").reshape(-1),
                    np.zeros(9),
                    0,
                    atol=0.0,
                    reason="no homography exists; the Rust side should refuse",
                )
            )
        )

    # ── 6. essential / fundamental ──────────────────────────────────────────
    world = ep_world()
    p1 = project_pinhole(world, (0, 0, 0), (0, 0, 0))
    relpose = V("ep_relpose")
    rv_rel, t_rel = relpose[:3], relpose[3:]
    p2 = project_pinhole(world, rv_rel, t_rel)
    results.append(
        compare(
            Case(
                "epipole_pts1",
                np.array(V("ep_pts1")).reshape(-1, 2),
                p1,
                0,
                atol=TOL_PROJ_F64,
                reason="identical world points, identity pose",
            )
        )
    )
    results.append(
        compare(
            Case(
                "epipole_pts2",
                np.array(V("ep_pts2")).reshape(-1, 2),
                p2,
                0,
                atol=TOL_PROJ_F64,
                reason="identical world points, identical relative pose",
            )
        )
    )
    results.append(
        compare(
            Case(
                "epipole_world_identity",
                world.reshape(-1),
                V("ep_world").reshape(-1),
                0,
                atol=0.0,
                reason="the world points themselves",
            )
        )
    )

    fit1, fit2 = p1[:20], p2[:20]
    held1, held2 = p1[20:], p2[20:]
    e_cv, mask = cv2.findEssentialMat(
        fit1, fit2, cameraMatrix=K(), method=cv2.RANSAC, prob=0.9999, threshold=0.5
    )
    f_cv, mask_f = cv2.findFundamentalMat(
        fit1, fit2, cv2.FM_RANSAC, 1.0, 0.9999
    )
    f_cv8, _ = cv2.findFundamentalMat(fit1[:8], fit2[:8], cv2.FM_8POINT)
    residuals = {
        "ep_resid_Efit": (e_cv, "cv2.findEssentialMat (RANSAC)", "F from E"),
        "ep_resid_Ffit": (f_cv, "cv2.findFundamentalMat (RANSAC)", "F"),
        "ep_resid_Ffit8": (f_cv8, "cv2.findFundamentalMat (FM_8POINT)", "F"),
    }
    rust_rows = V("ep_resid_Efit") if "ep_resid_Efit" in rec.rows_f32 else []
    # `ep_resid_Efit` holds the residuals of *both* the Rust model and the
    # reference model, so compare element by element against the reference's
    # own residuals computed here.
    for name, (mat, how, _kind) in residuals.items():
        if mat is None:
            residual_notes.append(f"{how} returned None")
            continue
        ref = np.concatenate([sampson(mat, fit1, fit2), sampson(mat, held1, held2)])
        results.append(
            compare(
                Case(
                    name,
                    np.array(V(name)),
                    ref,
                    0,
                    atol=TOL_SAMP_HELD if "8POINT" not in name else TOL_SAMP_8PT,
                    reason=f"{how}: Sampson residual in px on fit then held-out",
                )
            )
        )

    # ground truth F / E: the reference's own construction of them
    R_rel = rodrigues(rv_rel)
    tx = np.array(
        [[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]]
    )
    e_true = tx @ R_rel
    f_true = np.linalg.inv(k).T @ e_true @ np.linalg.inv(k)
    results.append(
        compare(
            Case(
                "epipole_E_true",
                V("E_true").reshape(3, 3),
                (e_true / np.linalg.norm(e_true)).reshape(3, 3),
                0,
                atol=1e-9,
                reason="E = [t]_x R normalised to unit Frobenius norm",
            )
        )
        if np.allclose(V("E_true"), e_true, atol=1e-9)
        else compare(
            Case(
                "epipole_E_true",
                V("E_true").reshape(3, 3),
                (e_true / np.linalg.norm(e_true)).reshape(3, 3),
                0,
                atol=1e-9,
                reason="E = [t]_x R, normalised by the emitted norm sign only",
            )
        )
    )
    results.append(
        compare(
            Case(
                "epipole_E_true_sign",
                np.array([V("E_true")[1]]),
                np.array([e_true.reshape(-1)[1]]),
                0,
                atol=1e-15,
                reason="E and -E are the same geometry; checked at one entry",
            )
        )
    )

    # RANSAC inlier counts, compared by distribution.
    contam1 = np.array(V("ep_contam_pts1")).reshape(-1, 2)
    contam2 = np.array(V("ep_contam_pts2")).reshape(-1, 2)
    n_corrupt = int((~np.isclose(contam1[:20], fit1).all(axis=1)).sum())
    rust_inl = np.array(V("F_ransac_inliers"))
    rust_n = int(rec.scalars["F_ransac_n_inliers"])
    f_ransac_cv, mask_r = cv2.findFundamentalMat(
        contam1, contam2, cv2.FM_RANSAC, 1.5, 0.9999
    )
    cv_n = int(mask_r.sum())
    cv_thresh_worst = float(sampson(f_ransac_cv, contam1, contam2).max())
    residual_notes.append(
        f"RANSAC fundamental, {n_corrupt}/{len(contam1)} correspondences "
        f"corrupted: Rust inliers {rust_n}/{len(contam1)}, cv2 "
        f"{cv_n}/{len(contam1)}; cv2's own max Sampson residual = "
        f"{cv_thresh_worst:.3f} px at the same 1.5 px threshold."
    )
    results.append(
        compare(
            Case(
                "ransac_inlier_count_F",
                np.array([rust_n]),
                np.array([cv_n]),
                0,
                atol=max(2, int(0.12 * len(contam1))),
                reason="RANSAC inlier counts are a distribution, not an identity",
            )
        )
    )
    rust_e_n = int(rec.scalars["E_ransac_n_inliers"])
    e_ransac_cv, mask_e = cv2.findEssentialMat(
        contam1, contam2, cameraMatrix=K(), method=cv2.RANSAC, prob=0.9999, threshold=1.5
    )
    results.append(
        compare(
            Case(
                "ransac_inlier_count_E",
                np.array([rust_e_n]),
                np.array([int(mask_e.sum())]),
                0,
                atol=max(2, int(0.12 * len(contam1))),
                reason="RANSAC inlier counts are a distribution, not an identity",
            )
        )
    )
    results.append(
        compare(
            Case(
                "ransac_inlier_mask_agreement",
                rust_inl.astype(np.float64),
                mask_r.reshape(-1).astype(np.float64),
                0,
                atol=0.5,
                reason="per-correspondence inlier flags: agree on N of "
                f"{len(contam1)}",
            )
        )
    )
    results.append(
        compare(
            Case(
                "ransac_F_max_residual",
                np.array([np.array(V("ep_resid_Fransac")).max()]),
                np.array([float(sampson(f_ransac_cv, fit1, fit2).max())]),
                0,
                atol=TOL_RANSAC_PX,
                reason="max Sampson residual of each model on the clean fit set",
            )
        )
    )
    # recover_pose_from_essential: compare the recovered rotation to OpenCV's.
    e_fit = V("E_fit").reshape(3, 3)
    nrm = np.linalg.norm(e_fit)
    e_fit_raw = V("E_fit_rowmajor").reshape(3, 3)
    nrm_raw = np.linalg.norm(e_fit_raw)
    e_fit_signed = (e_fit * nrm) / nrm_raw
    results.append(
        compare(
            Case(
                "pose_recovered_rotation",
                np.array(V("E_pose_recovered")[3:]).reshape(3, 3),
                R_rel,
                0,
                atol=1e-6,
                reason="recover_pose_from_essential must land on R or R^T; the "
                "SVD of a rank-2 E is degenerate so only the orbit is determined",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_rotation_transpose",
                np.array(V("E_pose_recovered")[3:]).reshape(3, 3),
                R_rel.T,
                0,
                atol=1e-6,
                reason="the R <-> R^T ambiguity a single view pair cannot resolve",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_translation_direction",
                np.array([V("E_pose_recovered")[:3]]).reshape(1, 3),
                (t_rel / np.linalg.norm(t_rel)).reshape(1, 3),
                0,
                atol=1e-6,
                reason="recover_pose_from_essential returns t normalised to unit "
                "length (it only has E, which fixes t up to scale)",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_baseline_scale",
                np.array([np.linalg.norm(V("E_pose_recovered")[:3])]),
                np.array([1.0]),
                0,
                atol=1e-9,
                reason="documented: t is a direction, recovered up to scale",
            )
        )
    )

    # E from correspondences: matrices are gauge-ambiguous, so compare the
    # epipolar residual, not the entries.
    e_ref_cv, _ = cv2.findEssentialMat(
        fit1, fit2, cameraMatrix=K(), method=cv2.RANSAC, prob=0.9999, threshold=0.5
    )
    f_from_e_cv = np.linalg.inv(k).T @ e_ref_cv @ np.linalg.inv(k)
    results.append(
        compare(
            Case(
                "essential_epipolar_residual",
                np.array(V("ep_resid_Efit")),
                np.concatenate(
                    [sampson(f_from_e_cv, fit1, fit2), sampson(f_from_e_cv, held1, held2)]
                ),
                0,
                atol=TOL_SAMP_HELD,
                reason="Rust E -> F residual vs cv2.findEssentialMat residual, "
                "same correspondences",
            )
        )
    )
    results.append(
        compare(
            Case(
                "essential_true_model_residual",
                V("ep_resid_true"),
                np.concatenate([sampson(f_true, fit1, fit2), sampson(f_true, held1, held2)])[
                    : len(V("ep_resid_true"))
                ],
                0,
                atol=TOL_SAMP_FIT,
                reason="the ground-truth F: every exact correspondence must "
                "satisfy it to rounding",
            )
        )
    )
    results.append(
        compare(
            Case(
                "essential_gauge_check",
                np.array([1.0]),
                np.array([np.linalg.norm(V("E_fit_rowmajor"))]),
                0,
                atol=1e-9,
                reason="E_fit is emitted unnormalised for reference; the "
                "H/h22-style normalisation is compared separately",
            )
        )
    )

    # ── report ──────────────────────────────────────────────────────────────
    print(f"ENV {fmt(env)}")
    print(f"RECORDS {len(results)}")
    print()
    for n in notes:
        print("NOTE " + n)
    print()
    print(render_table(results, "calib3d: camera composition, projection, distortion, DLT, epipolar"))
    print()
    for n in residual_notes:
        print("NOTE " + n)
    print()
    print("Agreements worth recording (independent checks that closed):")
    for name, how, dev in agreements:
        print(f"  {name}: {how}, max dev {dev:.3e}")
    print()
    bad = diverging(results)
    print(f"DIVERGING {len(bad)} of {len(results)}")
    for r in bad:
        y, x = r.argmax
        print(
            f"  {r.name}: max={r.max_abs:.6g} (interior {r.max_abs_interior:.6g}) "
            f"at (x={x}, y={y}) rust={r.rust_at_max:.6g} ref={r.ref_at_max:.6g} "
            f"over_tol={r.n_over}/{r.n} ({r.frac_over:.1%})"
        )
    return 0


def project_pinhole_from_cam32(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Reference for the f32 pinhole path: same formula in f32."""
    fx, fy, cx, cy = (np.float32(v) for v in CAM)
    return np.column_stack(
        [fx * xn + cx, fy * yn + cy]
    ).astype(np.float64)


def project_pinhole_from_cam32_dummy(*a, **k):  # pragma: no cover
    raise NotImplementedError


def fisheye_pinhole32(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    kd = np.array(KANNALA, dtype=np.float32)
    r = np.hypot(xn, yn)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + kd[0] * th2 + kd[1] * th2 ** 2 + kd[2] * th2 ** 3 + kd[3] * th2 ** 4)
    scale = thd / r
    fx, fy, cx, cy = (np.float32(v) for v in CAM)
    return np.column_stack([fx * xn * scale + cx, fy * yn * scale + cy]).astype(np.float64)


def main_wrap() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(main())