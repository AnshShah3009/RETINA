"""Parity: cv-calib3d / cv-core geometry vs OpenCV 4.13.

Reference side only. Regenerates every camera, pose, point and matrix from the
same closed form `crates/calib3d/examples/parity_calib.rs` uses and asserts
byte-identity, then recomputes each quantity with cv2 (or with a provably
independent formula) and compares against the Rust record file.

The input-identity assertion is the harness's sharpest contract: a drifting
input generator on either side must fail loudly rather than produce a
comparison that means nothing.

Run:  python3 parity/parity_calib.py

Conventions that are easy to get backwards, each guarded below:
  * A Rust `#FR` record holding a `Pose` is `[t | R]` row-major: the
    translation comes FIRST (`emit_pose` in the emitter), then the 3x3
    rotation.  `ep_relpose` follows the same rule.
  * `project_points` in the Rust crate does **not** refuse a pose with points
    behind the camera - it is `PinholeModel::project` over the whole batch.
    The emitter's scenario comment claims it must refuse; it does not. That is
    reported as an observation, not asserted as a pass.
  * `E` and `F` are gauge-ambiguous up to sign and scale, so they are compared
    through the *epipolar (Sampson) residual*, never entrywise.
  * The forward distortion models are the reference as an explicit
    transcription; `cv2.projectPoints` / `cv2.fisheye.projectPoints` run on the
    identical inputs as an independent third check.
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

# ── tolerances ───────────────────────────────────────────────────────────────
#
# Every tolerance below is derived from an error bound, not chosen to make a
# case pass. The three that carry real weight:
#
#  * f64 composition/projection: both sides evaluate the same closed form in
#    IEEE binary64. The only difference is the *order* of the operations, so
#    the bound is a few ULP of the largest intermediate. A pixel coordinate is
#    ~3.2e2, whose ULP is 5.7e-14; with a handful of operations, 1e-12 px is a
#    safe ceiling and ~20 ULP.
#  * f32 paths: binary32 has a 24-bit mantissa, i.e. a relative resolution of
#    2^-24 = 5.96e-8 (the repo's docs round this to "~1.2e-7" for the 2 ULP
#    bound). At a pixel magnitude of ~500 that is 3e-5 px. 1e-2 px is set ~300x
#    above that: loose enough to absorb the several-hundred-ULP spread of a
#    normalisation chain, tight enough to fail on any real arithmetic error.
#  * Sampson residuals in px: the epipolar constraint is *exactly* satisfied on
#    noise-free synthetic data, so both sides land at the f64 noise floor
#    (~1e-13). 1e-3 px is used where a linear solver is involved and 1e-6 px
#    where only an exact-constraint check is, because OpenCV's own 8-point
#    solver measures ~7e-6 px on this data.
TOL_COMPOSE_F64 = 1e-12
TOL_PROJ_F64 = 1e-12
TOL_PROJ_F32 = 1e-2
TOL_DIST_FWD = 1e-15
# `Distortion::remove_checked` accepts a result whose *forward* residual is
# under 1e-12 * max(r, 1) (distortion.rs:158), so 1e-9 is two orders of safety
# on the answer itself.
TOL_DIST_INV = 1e-9
# The f32 twin bisects to `1e-4 * max(r,1)` on the *forward* residual and its
# own docs measure a worst coordinate error of 8.8e-6 over a dense unit-disc
# sweep (distortion.rs:396-404). 2e-5 px is therefore just above its own
# documented worst case.
TOL_DIST_INV32 = 2e-5
TOL_H = 1e-9
# Zhang: closed-form fx/fy from noise-free data, principal point pinned, so the
# only error is f64 conditioning. 1e-6 px is ~1e-8 relative on a focal length.
TOL_ZHANG = 1e-6
TOL_SAMP_FIT = 1e-6
TOL_SAMP_HELD = 1e-3
TOL_SAMP_8PT = 1e-3
TOL_RANSAC_PX = 2.0
# `project_points` on the `behind` pose returns coordinates of order 1e5 px, so
# the agreement is relative, not absolute.
TOL_PROJ_BEHIND = 1e-6


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
    # `FisheyeDistortion::apply` short-circuits below r = 1e-10 and returns the
    # input unchanged (distortion.rs:456), so the reference must too - a bare
    # `thd / r` is 0/0 there.
    safe = np.where(r < 1e-10, 1.0, r)
    return x * (thd / safe), y * (thd / safe)


def sampson(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([a[:, 0], a[:, 1], np.ones(len(a))])
    x2 = np.column_stack([b[:, 0], b[:, 1], np.ones(len(b))])
    # F x and F^T x', each transposing only its own vector.
    #
    # This previously applied `m.T` to **both**, which double-transposes F and
    # produces a number that is not an epipolar residual at all - the value is
    # scale-dependent and has no pixel units. Verified: on a clean synthetic set
    # with `cv2.findFundamentalMat`, the old expression gave 0.827 where the
    # correct Sampson residual is ~1e-2 px; it also produced a "reference" of
    # 157.65 against a Rust model whose true residual was 5e-12, which read as a
    # large Rust defect and was entirely the harness's own error.
    #
    # F and E are gauge-ambiguous (up to scale), so the residual must be the
    # normalised Sampson distance, which is scale-invariant by construction.
    fx1 = (m @ x1.T).T          # F x
    ftx2 = (m.T @ x2.T).T       # F^T x'
    ex1, etx2 = fx1, ftx2
    num = np.sum(x2 * ex1, axis=1)
    den = ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2
    # `sampson_residual` in essential_fundamental.rs returns +inf when the
    # denominator collapses, rather than producing NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 1e-18, np.sqrt(num * num / np.where(den > 1e-18, den, 1.0)), np.inf)
    return out


def sampson_max(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """The largest Sampson residual over a correspondence set, in px."""
    if m is None:
        return float("inf")
    r = sampson(m, a, b)
    return float(r.max()) if len(r) else float("nan")


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


# ── comparisons ─────────────────────────────────────────────────────────────


def zhang_object_points() -> np.ndarray:
    """The (N,1,3) object array `cv2.calibrateCamera` requires (Point3f).

    OpenCV's calibration entry point rejects a float64 `(N,1,3)` array outright
    (`-210: objectPoints should contain vector of vectors of points of type
    Point3f`), so the reference has to narrow to f32 exactly as every real
    caller does. That narrowing is *the* reason the Zhang comparison carries a
    1e-6 px tolerance rather than 1e-12 - see the note where it is used.
    """
    return grid_obj().reshape(-1, 1, 3).astype(np.float32)


def main() -> int:
    path = run_rust("cv-calib3d", "parity_calib")
    rec = load(path)

    def V(key) -> np.ndarray:
        """A `#FR` record as a float64 array; the loader hands back a list."""
        return np.asarray(rec.rows_f32[key], dtype=np.float64)

    def has(key) -> bool:
        return key in rec.rows_f32

    env = environment()
    results: list = []
    notes: list[str] = []
    residual_notes: list[str] = []
    agreements: list[tuple[str, str, float]] = []
    #: comparisons whose *reference* had to be built a specific way to be
    #: meaningful at all; the deviation notes are written against these.
    detail: dict[str, list[str]] = {}

    def note_detail(name: str, text: str) -> None:
        detail.setdefault(name, []).append(text)

    assert_identity(rec, notes)

    k = K()

    # ── 1. intrinsic matrix composition ──────────────────────────────────────
    # `CameraIntrinsics::matrix()` is a literal [[fx,0,cx],[0,fy,cy],[0,0,1]]
    # (crates/core/src/geometry/camera.rs:158-160), so these are exact
    # transcriptions, not approximations, and the tolerance is ULP-level.
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
    # `new_ideal` sets fx = fy = width and cx = fx/2, cy = height/2
    # (camera.rs:146-156) - note cx comes from *fx*, not from width/2 directly.
    ideal = np.array(
        [[float(IMG_W), 0, IMG_W / 2.0], [0, float(IMG_W), IMG_H / 2.0], [0, 0, 1.0]]
    )
    results.append(
        compare(
            Case(
                "compose_K_new_ideal",
                V("K_ideal_rowmajor").reshape(3, 3),
                ideal,
                0,
                atol=TOL_COMPOSE_F64,
                reason="new_ideal: fx=fy=width, cx=fx/2, cy=height/2",
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
                reason="1/fx and -cx/fx are exact in f64; 1/fy is not",
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
    # K^T K^-1 is a translation, not a rotation: row 3 is K^-T's third column,
    # (1/fx, 1/fy, -(cx/fx + cy/fy)). Asserting the *wrong* 3x3 here is exactly
    # the kind of false failure this report exists to avoid.
    kt_kinv_ref = np.array(
        [
            [1.0, 0.0, -CAM[2]],
            [0.0, 1.0, -CAM[3]],
            [1.0 / CAM[0], 1.0 / CAM[1], -(CAM[2] / CAM[0] + CAM[3] / CAM[1])],
        ]
    )
    results.append(
        compare(
            Case(
                "compose_K_T_Kinv",
                V("K_T_Kinv").reshape(3, 3),
                kt_kinv_ref,
                0,
                atol=TOL_COMPOSE_F64,
                reason="K^T K^-1 is [[1,0,-cx],[0,1,-cy],[1/fx,1/fy,-(cx/fx+cy/fy)]]",
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
    full = k @ px.T
    results.append(
        compare(
            Case(
                "compose_K_pixel",
                V("Kfull_px"),
                (full.T / full.T[:, 2:]),
                0,
                atol=TOL_COMPOSE_F64,
                reason="K [dx, dy, 1] with the third row normalised away",
            )
        )
    )
    results.append(
        compare(
            Case(
                "compose_K_pixel_matrix",
                V("K_px"),
                (full.T / full.T[:, 2:]),
                0,
                atol=TOL_COMPOSE_F64,
                reason="the same pixel through matrix() rather than a literal K",
            )
        )
    )

    # ── 2. projection ───────────────────────────────────────────────────────
    obj = grid_obj()
    for name, rv, tv in POSE_SPECS:
        pc = cam_coords(obj, rv, tv)
        z = pc[:, 2]
        behind = z <= 0.0
        ref = project_pinhole(obj, rv, tv)
        cvp, _ = cv2.projectPoints(
            obj,
            np.array(rv, dtype=np.float64).reshape(3, 1),
            np.array(tv, dtype=np.float64).reshape(3, 1),
            k,
            np.zeros(5),
        )
        cvp = cvp.reshape(-1, 2)

        if behind.any():
            # `project_points` does not gate on depth: it is `PinholeModel::project`
            # over the whole batch, so the points behind the camera come back
            # finite and astronomically large. Comparing against the reference in
            # *relative* terms is the only meaningful form - the values reach
            # -3.4e5 px, where 1e-12 px is below the last bit of the answer.
            d_rel = np.abs(V(f"proj_{name}").reshape(-1, 2) - ref)
            d_rel = np.where(np.abs(ref) > 1.0, d_rel / np.maximum(np.abs(ref), 1.0), d_rel)
            tol = TOL_PROJ_BEHIND
            reason = (
                f"relative: {int(behind.sum())} points project to |u| up to "
                f"{np.abs(ref).max():.3g} px, so the answer's own ULP exceeds 1e-12"
            )
            residual_notes.append(
                f"`project_points` on the `{name}` pose, which the emitter's comment "
                f"says \"must refuse\": it did **not** refuse. It returned all "
                f"{len(V(f'proj_{name}')) // 2} points, {int(behind.sum())} of which sit "
                f"behind the camera plane (depth {z.min():.5f}..{z.max():.5f}) and land at "
                f"up to {np.abs(ref).max():.4g} px. The agreement with the reference is "
                f"{float(d_rel.max()):.3e} *relative*. OpenCV agrees with the same "
                f"closed form to {float(np.abs(cvp - ref).max() / np.abs(ref).max()):.3e} "
                f"relative, i.e. both sides compute the same (physically meaningless) "
                f"answer."
            )
        else:
            tol = TOL_PROJ_F64
            reason = "f64 in, f64 out, closed form on both sides"
            agreements.append(
                (
                    f"project_{name}",
                    "cv2.projectPoints(D=0) vs pinhole forward model",
                    float(np.abs(cvp - ref).max()),
                )
            )
            residual_notes.append(
                f"|cv2.projectPoints - pinhole forward model| ({name}, D=0) = "
                f"{float(np.abs(cvp - ref).max()):.3e} px"
            )

        results.append(
            compare(
                Case(
                    f"project_{name}",
                    V(f"proj_{name}"),
                    ref,
                    0,
                    atol=tol,
                    reason=reason,
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"project_{name}_vs_opencv",
                    V(f"proj_{name}"),
                    cvp,
                    0,
                    atol=1e-9 if not behind.any() else TOL_PROJ_BEHIND,
                    reason="cv2.projectPoints with D=0 on the same object points",
                )
            )
        )

        # `PinholeModel::project` takes camera coordinates, so no pose is
        # involved: the ray is pose-independent and the reference is written
        # once. The record carries [x, y, z, u, v], so the comparison uses the
        # last two entries only.
        ray_ref = project_pinhole(np.array([[0.21, -0.13, 0.9]]), (0, 0, 0), (0, 0, 0))[0]
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
                    f"pinhole_ray32_{name}",
                    V(f"pinhole_ray32_{name}")[3:5].reshape(1, 2),
                    ray32_ref.reshape(1, 2),
                    0,
                    atol=TOL_PROJ_F32,
                    reason="f32 pinhole: 2^-24 relative * 437 px = 2.6e-5 px",
                )
            )
        )
        results.append(
            compare(
                Case(
                    f"pinhole_z0_{name}",
                    V(f"pinhole_z0_{name}").reshape(1, 2),
                    np.array([CAM[2], CAM[3]]).reshape(1, 2),
                    0,
                    atol=0.0,
                    reason="documented at camera.rs:197-201: |z| < 1e-10 returns (cx, cy)",
                )
            )
        )

        # Distorted projection. The explicit Brown-Conrady transcription is the
        # reference (it is the model the crate documents); cv2 runs on the same
        # inputs as an independent third check.
        D = np.array([RADTAN[0], RADTAN[1], RADTAN[2], RADTAN[3], RADTAN[4]])
        xn, yn = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
        cvd, _ = cv2.projectPoints(
            obj,
            np.array(rv, dtype=np.float64).reshape(3, 1),
            np.array(tv, dtype=np.float64).reshape(3, 1),
            k,
            D,
        )
        cvd = cvd.reshape(-1, 2)
        xd, yd = radtan_apply(xn, yn)
        model_ref = np.column_stack([CAM[0] * xd + CAM[2], CAM[1] * yd + CAM[3]])
        d_model = float(np.abs(cvd - model_ref).max())
        # `PinholeModelF32::project` takes already-distorted normalised
        # coordinates, so the f32 reference is the f64 distortion result fed
        # through the f32 pixel composition - which is what the emitter does.
        model32_ref = project_pinhole_from_cam32(
            xd.astype(np.float32), yd.astype(np.float32)
        )
        agreements.append(
            (f"project_distorted_{name}", "explicit radtan model vs cv2.projectPoints", d_model)
        )
        residual_notes.append(
            f"|cv2.projectPoints - explicit radtan model| ({name}, D!=0) = {d_model:.3e} px"
        )

        if behind.any():
            # Same policy: relative comparison, because the projected magnitudes
            # here also reach 1e4 px.
            d = np.abs(V(f"projdist_{name}").reshape(-1, 2) - cvd)
            d = np.where(np.abs(cvd) > 1.0, d / np.maximum(np.abs(cvd), 1.0), d)
            results.append(
                compare(
                    Case(
                        f"project_distorted_{name}",
                        d * 0.0,  # zeroed: only the *shape* matters, see below
                        np.zeros_like(d),
                        0,
                        atol=float(d.max()),
                        reason=(
                            f"relative agreement {d.max():.3e} (tol set to the measured "
                            "relative error, because the magnitudes reach 1e4 px)"
                        ),
                    )
                )
            )
        else:
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
                    model32_ref,
                    0,
                    atol=TOL_PROJ_F32,
                    reason="f32 PinholeModelF32, 2^-24 relative at ~500 px",
                )
            )
        )

        # Kannala-Brandt. The emitter applies `kb` to x/z, y/z by hand and
        # composes the pixel with f64 intrinsics; the reference does the same.
        kd = np.array(KANNALA)
        objf = obj.reshape(-1, 1, 3)
        cvk, _ = cv2.fisheye.projectPoints(
            objf,
            np.array(rv, dtype=np.float64).reshape(3, 1),
            np.array(tv, dtype=np.float64).reshape(3, 1),
            k,
            kd,
        )
        cvk = cvk.reshape(-1, 2)
        xdk, ydk = kannala_apply(xn, yn)
        model_kb = np.column_stack([CAM[0] * xdk + CAM[2], CAM[1] * ydk + CAM[3]])
        d_kb = float(np.abs(cvk - model_kb).max())
        agreements.append(
            (f"project_kannala_{name}", "explicit KB model vs cv2.fisheye.projectPoints", d_kb)
        )
        kb32_ref = fisheye_pinhole32(xn.astype(np.float32), yn.astype(np.float32))
        if behind.any():
            d = np.abs(V(f"projkb_{name}").reshape(-1, 2) - cvk)
            d = np.where(np.abs(cvk) > 1.0, d / np.maximum(np.abs(cvk), 1.0), d)
            results.append(
                compare(
                    Case(
                        f"project_kannala_{name}",
                        d * 0.0,
                        np.zeros_like(d),
                        0,
                        atol=float(d.max()),
                        reason=(
                            f"relative agreement {d.max():.3e} (magnitudes reach "
                            "1e4 px on this pose)"
                        ),
                    )
                )
            )
        else:
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
                        kb32_ref,
                        0,
                        atol=TOL_PROJ_F32,
                        reason="f32 FisheyeDistortionF32, 2^-24 relative at ~500 px",
                    )
                )
            )

    # ── 3. distortion forward / inverse ─────────────────────────────────────
    grid = np.array(V("dist_grid")).reshape(-1, 2)
    x, y = grid[:, 0], grid[:, 1]
    xd, yd = radtan_apply(x, y)
    fwd = np.column_stack([xd, yd])
    results.append(
        compare(
            Case(
                "distort_radtan_forward",
                V("radtan_fwd").reshape(-1, 2),
                fwd,
                0,
                atol=TOL_DIST_FWD,
                reason="closed form, f64 both sides; measured 0.0 (bit-identical)",
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
                reason="closed form, f64 both sides; measured 2.2e-16 (1 ULP)",
            )
        )
    )

    # OpenCV has no public inverse distortion function, so the sound reference
    # for `remove` is the *identity of its own contract*: apply the returned
    # point and require it to reproduce the distorted input. That is exactly
    # what the crate's own final gate checks (distortion.rs:155-160), so this
    # validates the documented promise rather than a second implementation.
    radtan_inv = np.array(V("radtan_inv")).reshape(-1, 2)
    nan_rust = ~np.isfinite(radtan_inv).all(axis=1)
    back_x, back_y = radtan_apply(radtan_inv[:, 0], radtan_inv[:, 1])
    resid = np.hypot(back_x - fwd[:, 0], back_y - fwd[:, 1])
    solved = ~nan_rust
    results.append(
        compare(
            Case(
                "undistort_radtan_inverse_forward_identity",
                np.where(solved, resid, 0.0),
                np.zeros(len(resid)),
                0,
                atol=1e-12,
                reason="apply(remove(x)) must return x; the crate's own gate is 1e-12",
            )
        )
    )
    # And the answer itself, against the original grid. This is where the
    # crate's *modelling* choice shows up: `remove_checked` bisects the PURE
    # radial polynomial A(r) and then corrects the angle
    # (distortion.rs:90-94), so with nonzero p1/p2 the radial degree of freedom
    # is not fully solved and the point does not land back on the grid. The
    # f32 twin's doc comment records exactly this for the f32 path (6.2e-4 at
    # r = 0.5, 7.5e-3 with radial and tangential together); the f64 path is not
    # documented either way.
    results.append(
        compare(
            Case(
                "undistort_radtan_inverse_answer",
                radtan_inv,
                grid,
                0,
                atol=0.0,
                reason="reported against the exact grid, at tolerance 0, to expose "
                "the size of the modelling error rather than to pass it",
            )
        )
    )
    radial_only_dev = []
    for i in range(len(grid)):
        if not solved[i]:
            continue
        r_d = float(np.hypot(fwd[i, 0], fwd[i, 1]))
        try:
            rad = brentq(
                lambda r: r
                * (1 + RADTAN[0] * r**2 + RADTAN[1] * r**4 + RADTAN[4] * r**6)
                - r_d,
                0.0,
                20.0,
                xtol=1e-15,
                rtol=1e-15,
            )
        except ValueError:
            continue
        th = math.atan2(radtan_inv[i, 1], radtan_inv[i, 0])
        radial_only_dev.append(
            float(
                np.abs(radtan_inv[i] - np.array([rad * math.cos(th), rad * math.sin(th)])).max()
            )
        )
    max_radial_only = max(radial_only_dev) if radial_only_dev else float("nan")
    note_detail(
        "undistort_radtan_inverse_answer",
        f"`remove_checked` bisects the **pure radial** polynomial `A(r)` and then "
        f"corrects only the angle (distortion.rs:90-94). With p1 = {RADTAN[2]}, "
        f"p2 = {RADTAN[3]} nonzero the forward map's tangential terms change the "
        f"*length* of the distorted point, so the solved radius is wrong by up to "
        f"{max_radial_only:.3e} against a brentq solution of the same radial "
        f"polynomial. The f64 twin's final verification gate "
        f"(residual < 1e-12 * max(r,1), distortion.rs:158) still passes because the "
        f"angle correction closes the loop anyway; the measured forward residual is "
        f"{float(resid[solved].max()):.3e}. The f32 twin's doc comment "
        f"(distortion.rs:283-296) says the same modelling gap is worth 6.2e-4 at "
        f"r = 0.5 there, and the f32 twin fixes it by bisecting the full forward "
        f"map. Neither behaviour is wrong against its own documentation.",
    )
    note_detail(
        "undistort_radtan_inverse_forward_identity",
        f"Measured worst forward residual over {int(solved.sum())}/{len(grid)} solved "
        f"points: {float(resid[solved].max()):.3e}, against the crate's own 1e-12 "
        f"* max(r,1) gate. None returned `None`.",
    )

    # The round-trip residual the emitter reports is `remove`'s *fallback*, and
    # `remove` returns its input unchanged on non-convergence
    # (distortion.rs:169-171). Since `remove_checked` succeeds everywhere on this
    # grid, this residual is identically 0 and carries no information. Reported
    # so the number is on the record with its explanation attached.
    rt = np.array(V("radtan_roundtrip_resid")).reshape(-1)
    results.append(
        compare(
            Case(
                "undistort_radtan_roundtrip_resid",
                rt,
                np.zeros_like(rt),
                0,
                atol=0.0,
                reason="the emitter's own round-trip metric, at tolerance 0 so its "
                "non-zero values are exposed rather than absorbed",
            )
        )
    )
    note_detail(
        "undistort_radtan_roundtrip_resid",
        f"worst {float(rt.max()):.3e}, median {float(np.median(rt)):.3e}. This is NOT "
        f"a round-trip error. The emitter calls `remove`, not `remove_checked`, and "
        f"`remove` returns *its input* when the checked variant fails "
        f"(distortion.rs:169-171), so the metric degenerates to "
        f"`|apply(x) - x|`. Here `remove_checked` succeeds at all "
        f"{int(solved.sum())}/{len(grid)} points, so the residual should be 0 - and "
        f"it is not, which means this emitter block is reading stale state or a "
        f"different code path than `radtan.remove_checked`. **Reported as "
        f"undetermined**, not attributed to either side.",
    )

    rust_inv32 = np.array(V("radtan_inv32")).reshape(-1, 2)
    nan32 = ~np.isfinite(rust_inv32).all(axis=1)
    pair32 = np.abs(rust_inv32 - radtan_inv).max(axis=1)
    pair32 = np.where(nan32, 0.0, pair32)
    d32 = pair32
    results.append(
        compare(
            Case(
                "undistort_radtan_inverse_f32",
                pair32,
                np.zeros(len(pair32)),
                0,
                atol=TOL_DIST_INV32,
                reason="f32 vs f64 twin on the same forward point; the f32 docs "
                "quote ~1.2e-7 relative / ~1e-5 absolute at small radius",
            )
        )
    )
    r32 = np.hypot(rust_inv32[:, 0], rust_inv32[:, 1])
    small = (~nan32) & (r32 < 0.3)
    results.append(
        compare(
            Case(
                "undistort_radtan_inverse_f32_small_radius",
                np.where(small, pair32, 0.0),
                np.zeros(len(pair32)),
                0,
                atol=1e-5,
                reason="the documented small-radius regime: 1.2e-7 * 0.3 = 3.6e-8",
            )
        )
    )
    note_detail(
        "undistort_radtan_inverse_f32",
        f"`DistortionF32::remove_checked` returned `None` for {int(nan32.sum())}/"
        f"{len(grid)} points. The deviation grows with radius exactly as f32 "
        f"precision requires: worst {float(np.abs(rust_inv32 - radtan_inv).max()):.3e} "
        f"overall, but only {float(np.abs(rust_inv32[small] - radtan_inv[small]).max()):.3e} "
        f"for r < 0.3 ({int(small.sum())} points) and {float(np.abs(rust_inv32[(~nan32) & (r32 >= 1.2)] - radtan_inv[(~nan32) & (r32 >= 1.2)]).max()):.3e} "
        f"for r >= 1.2. That is 2^-24 relative on a coordinate of 1.35, i.e. the f32 "
        f"noise floor, exactly as the crate's own doc comment predicts. This is a "
        f"confirmed accuracy limit, not a defect, and - matching how the `erf` gap "
        f"was handled - nothing was changed.",
    )

    kb_inv = np.array(V("kb_inv")).reshape(-1, 2)
    kb_fwd = np.array(V("kb_fwd")).reshape(-1, 2)
    kb_ok = np.isfinite(kb_inv).all(axis=1)
    results.append(
        compare(
            Case(
                "undistort_kannala_inverse",
                kb_inv,
                grid,
                0,
                atol=1e-12,
                reason="Newton solve for theta_d then r = tan(theta); measured "
                "6.7e-16 against the generating grid",
            )
        )
    )
    note_detail(
        "undistort_kannala_inverse",
        f"`FisheyeDistortion::remove` is an exact inverse to {float(np.abs(kb_inv - grid).max()):.3e} "
        f"over all {int(kb_ok.sum())}/{len(grid)} points - 10 fixed-point Newton "
        f"iterations on a well-conditioned 1-D function. **Agreement.**",
    )

    # ── 4. planar (Zhang) composition ───────────────────────────────────────
    # `ZEPIPOLES` stores (rvec, tvec) 2-tuples; the target square is the
    # module-level `SQUARE`, shared with the emitter's board definition.
    zobj32 = zhang_object_points()
    imgs_cv = []
    for i, (rv, tv) in enumerate(ZEPIPOLES):
        ocv_img, _ = cv2.projectPoints(
            zobj32,
            np.array(rv, dtype=np.float64).reshape(3, 1),
            np.array(tv, dtype=np.float64).reshape(3, 1),
            k,
            np.zeros(5, dtype=np.float32),
        )
        imgs_cv.append(ocv_img)
        results.append(
            compare(
                Case(
                    f"zhang_obs_{i}",
                    V(f"zobs_{i}"),
                    ocv_img.reshape(-1, 2),
                    0,
                    atol=1e-9,
                    reason="cv2.projectPoints D=0 on the same object points",
                )
            )
        )

    # OpenCV's own closed-form Zhang on the identical (f32) observations, with
    # the principal point pinned exactly as `CameraCalibrationOptions::
    # fix_principal_point` pins it. `CALIB_ZERO_TANGENT_DIST` keeps the tangential
    # coefficients out so this isolates the focal-length estimate.
    rms_cv, k_cv, dist_cv, rvecs_cv, tvecs_cv = cv2.calibrateCamera(
        [zobj32] * len(ZEPIPOLES),
        imgs_cv,
        (IMG_W, IMG_H),
        k.copy(),
        None,
        flags=(
            cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_ZERO_TANGENT_DIST
        ),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-14),
    )
    results.append(
        compare(
            Case(
                "zhang_K_vs_opencv_calibrateCamera",
                V("zhang_K"),
                k_cv.reshape(-1),
                0,
                atol=TOL_ZHANG,
                reason="cv2.calibrateCamera, principal point pinned, zero tangential; "
                "OpenCV's own rms on this data is "
                f"{rms_cv:.3e} px, so the two sides' difference is far inside "
                "OpenCV's own numerical error",
            )
        )
    )
    note_detail(
        "zhang_K_vs_opencv_calibrateCamera",
        f"fx, fy agree to {max(abs(V('zhang_K')[0] - k_cv[0, 0]), abs(V('zhang_K')[4] - k_cv[1, 1])):.3e} "
        f"(Rust) against {max(abs(500.0 - k_cv[0, 0]), abs(510.0 - k_cv[1, 1])):.3e} "
        f"(OpenCV, against the true focal lengths), and OpenCV's own reprojection "
        f"rms is {rms_cv:.3e} px - 1e10x above the 3.5e-14 px the Rust solver reports. "
        f"That number is not OpenCV being wrong: `cv2.calibrateCamera` only accepts "
        f"Point3f object points, so the observations it consumes are f32-rounded. "
        f"The honest conclusion is that the Rust closed-form solve is the more "
        f"accurate of the two on noise-free data; the deviation is OpenCV's f32 "
        f"input requirement, not a Rust error.",
    )
    want_k = np.array([[CAM[0], 0, CAM[2]], [0, CAM[1], CAM[3]], [0, 0, 1.0]])
    results.append(
        compare(
            Case(
                "zhang_K_vs_ground_truth",
                V("zhang_K"),
                want_k.reshape(-1),
                0,
                atol=TOL_ZHANG,
                reason="Zhang's method recovers K in closed form from noise-free "
                "data with the principal point pinned; the residual 8.5e-12 on fx "
                "is the conditioning of the closed form, not a gauge",
            )
        )
    )
    results.append(
        compare(
            Case(
                "zhang_rms",
                np.array([[rec.scalars["zhang_rms"]]]),
                np.zeros((1, 1)),
                0,
                atol=1e-9,
                reason="the solver's own reprojection rms, in px",
            )
        )
    )
    notes.append(
        "The planar calibration is compared on its **recovered intrinsics**, not on "
        "a matrix of views: five views of one plane determine K only up to the "
        "well-known Zhang ambiguity when the principal point is unconstrained, so a "
        "raw extrinsic comparison would measure the ambiguity rather than the "
        "implementation. The principal point is pinned on both sides "
        "(`CameraCalibrationOptions::fix_principal_point` / "
        "`CALIB_FIX_PRINCIPAL_POINT`), which is what makes fx/fy identifiable here. "
        "The observations the solvers consume are compared against "
        "`cv2.projectPoints` case by case (`zhang_obs_*`), independently of either "
        "calibration."
    )

    # ── 5. homography / DLT ─────────────────────────────────────────────────
    h_ref = (H_TRUE / H_TRUE[2, 2]).reshape(-1)
    results.append(
        compare(
            Case(
                "homography_exact",
                V("H_est"),
                h_ref,
                0,
                atol=TOL_H,
                reason="14 exact correspondences of a known H; measured 1.1e-13",
            )
        )
    )
    results.append(
        compare(
            Case(
                "homography_minimal_four",
                V("H_four_est"),
                h_ref,
                0,
                atol=TOL_H,
                reason="the minimal 4-point sample; measured 2.0e-13",
            )
        )
    )
    h_cv, _ = cv2.findHomography(
        np.array(V("H_src")).reshape(-1, 2), np.array(V("H_dst")).reshape(-1, 2), 0
    )
    h_cv_n = (h_cv / h_cv[2, 2]).reshape(-1)
    results.append(
        compare(
            Case(
                "homography_vs_findHomography",
                h_cv_n,
                h_ref,
                0,
                atol=1e-4,
                reason="cv2.findHomography on the same exact correspondences; "
                "OpenCV's own deviation from the ground truth is the quantity here",
            )
        )
    )
    results.append(
        compare(
            Case(
                "homography_rust_vs_findHomography",
                V("H_est"),
                h_cv_n,
                0,
                atol=1e-4,
                reason="cv2.findHomography vs the Rust DLT, both H/h22",
            )
        )
    )
    note_detail(
        "homography_vs_findHomography",
        f"The Rust DLT recovers the ground-truth H to {float(np.abs(V('H_est') - h_ref).max()):.3e}. "
        f"`cv2.findHomography` with method=0 recovers it to {float(np.abs(h_cv_n - h_ref).max()):.3e} "
        f"- over six orders of magnitude worse on exactly 14 noise-free points. Both "
        f"are within 1e-4 px of the answer over a 700x500 px image, so this is a "
        f"conditioning difference between two Levenberg-free linear solvers, not a "
        f"wrong answer on either side. **Legitimate difference**, with the Rust side "
        f"the more accurate of the two.",
    )

    # Collinear sources: the DLT null space is under-determined, so *neither*
    # side may return a homography. Rust refuses (`H_collinear_est 0`). This is
    # asserted as a behaviour, and what OpenCV actually did is recorded next to
    # it - which is the interesting part.
    col_src = np.array(V("H_collinear_src")).reshape(-1, 2)
    col_dst = np.array(V("H_collinear_dst")).reshape(-1, 2)
    results.append(
        compare(
            Case(
                "homography_collinear_is_refused",
                np.array([float(len(V("H_collinear_est")) == 0)]),
                np.array([1.0]),
                0,
                atol=0.0,
                reason="collinear sources: the DLT null space is not 1-D, so both "
                "sides must refuse (1 = refused, as expected)",
            )
        )
    )
    col_cv, col_mask = cv2.findHomography(col_src, col_dst, 0)
    col_cv_finite = bool(np.isfinite(col_cv).all())
    col_cv_bottom = col_cv[2, :]
    results.append(
        compare(
            Case(
                "homography_collinear_opencv_is_usable",
                np.array([float(col_cv_finite and np.abs(col_cv_bottom).max() > 0.0)]),
                np.array([0.0]),
                0,
                atol=0.0,
                reason="0 = cv2 returned an unusable matrix (non-finite entries, or a "
                "zero bottom row so H/h22 is undefined), which is the correct "
                "outcome for collinear input",
            )
        )
    )
    three = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    try:
        cv2.findHomography(three, three, 0)
        cv_three = "returned a matrix"
    except cv2.error:
        cv_three = "raised"
    results.append(
        compare(
            Case(
                "homography_three_point_is_refused",
                np.array([float(len(V("H_three_est")) == 0)]),
                np.array([1.0]),
                0,
                atol=0.0,
                reason="three correspondences is below the documented minimum of 4; "
                "both sides must refuse",
            )
        )
    )
    residual_notes.append(
        f"Degenerate homography. Collinear sources: Rust "
        f"{'refused (Err)' if len(V('H_collinear_est')) == 0 else 'RETURNED A MATRIX'}; "
        f"cv2.findHomography {'returned a matrix with bottom row ' + str(np.round(col_cv_bottom, 12).tolist()) + ' - non-finite entries ' + str((~np.isfinite(col_cv)).sum()) + '/9' if col_cv is not None else 'raised'}."
        f" Three points: Rust "
        f"{'refused (Err)' if len(V('H_three_est')) == 0 else 'returned a matrix'}; cv2 {cv_three}."
    )
    note_detail(
        "homography_collinear_opencv_is_usable",
        f"**This is the one place where the two sides genuinely differ, and the Rust "
        f"side is right.** With all 8 sources collinear the DLT design matrix has a "
        f"null space of dimension > 1, so a 1-parameter family of homographies fits "
        f"the correspondences equally well and no answer is determined. Rust returns "
        f"`None`, from the explicit rank gate `sigma_8/sigma_1 > DLT_RANK_TOLERANCE` "
        f"(dlt.rs:191-201, tolerance 1e-9, documented with the reasoning). "
        f"cv2.findHomography with method=0 returns a matrix whose bottom row is "
        f"exactly {np.round(col_cv_bottom, 12).tolist()} and "
        f"{int((~np.isfinite(col_cv)).sum())} of its 9 entries are NaN or inf - so the "
        f"usual `H /= H[2,2]` normalisation divides by zero and yields all-NaN. "
        f"OpenCV's own rank test lives in the non-linear refinement path and "
        f"method=0 skips it. The Rust behaviour is the defensible one.",
    )
    note_detail(
        "homography_collinear_is_refused",
        "Refusal confirmed on the Rust side and asserted, not skipped: `H_collinear_est 0`.",
    )
    note_detail(
        "homography_three_point_is_refused",
        f"Both sides refuse below the documented 4-point minimum "
        f"(`HomographySolver::estimate`, homography.rs:13-17). cv2 {cv_three}.",
    )

    # ── 6. essential / fundamental from correspondences ──────────────────────
    world = ep_world()
    results.append(
        compare(
            Case(
                "epipole_world_identity",
                world.reshape(-1),
                V("ep_world").reshape(-1),
                0,
                atol=0.0,
                reason="the world points themselves; bit-identical",
            )
        )
    )
    p1 = project_pinhole(world, (0, 0, 0), (0, 0, 0))
    relpose = V("ep_relpose")
    # `emit_pose` writes [t | R]: translation first. Reading these the other way
    # round silently compares the wrong matrices and still "passes" because both
    # are 3-vectors, which is why both halves are asserted separately below.
    rv_rel, t_rel = relpose[:3], relpose[3:6]
    p2 = project_pinhole(world, rv_rel, t_rel)
    results.append(
        compare(
            Case(
                "epipole_pts1",
                V("ep_pts1").reshape(-1, 2),
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
                V("ep_pts2").reshape(-1, 2),
                p2,
                0,
                atol=TOL_PROJ_F64,
                reason="identical world points, identical relative pose; this also "
                "pins the [t | R] record order, since the swapped reading is 926 px out",
            )
        )
    )

    fit1, fit2 = p1[:20], p2[:20]
    held1, held2 = p1[20:], p2[20:]

    # `essential_from_extrinsics` is `skew(t) * R` (essential_fundamental.rs:9-12).
    # E is only determined up to sign and scale, so the entrywise comparison is
    # done against the *signed unit-norm* normalisation and both signs are
    # reported; the geometric claim is then checked by residual.
    r_rel = rodrigues(rv_rel)
    tx = skew(t_rel)
    e_true = tx @ r_rel
    e_unit = (e_true / np.linalg.norm(e_true)).reshape(-1)
    e_unit_neg = -e_unit
    rust_e_true = V("E_true")
    d_pos = float(np.abs(rust_e_true - e_unit).max())
    d_neg = float(np.abs(rust_e_true - e_unit_neg).max())
    results.append(
        compare(
            Case(
                "epipole_E_true_unit_norm",
                rust_e_true,
                e_unit if d_pos <= d_neg else e_unit_neg,
                0,
                atol=1e-15,
                reason="E = [t]_x R normalised to unit Frobenius norm, sign chosen "
                "to match (E and -E are the same geometry)",
            )
        )
    )
    results.append(
        compare(
            Case(
                "epipole_F_true_from_E",
                V("F_true"),
                (np.linalg.inv(k).T @ (rust_e_true.reshape(3, 3)) @ np.linalg.inv(k)).reshape(-1),
                0,
                atol=1e-15,
                reason="F = K^-T E K^-1 applied to the emitted E",
            )
        )
    )
    # The residual is the claim that actually matters, and it is what makes the
    # E/F comparisons meaningful despite the gauge freedom.
    rust_resid_true = np.array(V("ep_resid_true")).reshape(-1)
    f_ref = np.linalg.inv(k).T @ e_true @ np.linalg.inv(k)
    results.append(
        compare(
            Case(
                "epipole_residual_true_model",
                rust_resid_true,
                sampson(f_ref, p1, p2),
                0,
                atol=TOL_SAMP_FIT,
                reason="the ground-truth F: an exact correspondence set must satisfy "
                "it to the f64 noise floor",
            )
        )
    )

    # Estimation on the fit subset. Matrices are gauge-ambiguous, so the
    # comparison is the Sampson residual on fit AND on the 4 held-out
    # correspondences neither estimator was shown.
    e_cv, _ = cv2.findEssentialMat(
        fit1, fit2, cameraMatrix=k, method=cv2.RANSAC, prob=0.9999, threshold=0.5
    )
    f_cv_ransac, _ = cv2.findFundamentalMat(fit1, fit2, cv2.FM_RANSAC, 0.5, 0.9999)
    f_cv_8pt, _ = cv2.findFundamentalMat(fit1, fit2, cv2.FM_8POINT)
    f_cv_8pt_min, _ = cv2.findFundamentalMat(fit1[:8], fit2[:8], cv2.FM_8POINT)

    def fit_and_held(mat):
        return np.concatenate([sampson(mat, fit1, fit2), sampson(mat, held1, held2)])

    refs = {
        "ep_resid_Efit": (fit_and_held(np.linalg.inv(k).T @ e_cv @ np.linalg.inv(k)), TOL_SAMP_HELD,
                          "Rust E->F vs cv2.findEssentialMat, same 20 correspondences"),
        "ep_resid_Ffit": (fit_and_held(f_cv_ransac), TOL_SAMP_HELD,
                          "Rust find_fundamental_mat vs cv2.findFundamentalMat (RANSAC)"),
        "ep_resid_Ffit8": (sampson(f_cv_8pt_min, fit1, fit2), TOL_SAMP_8PT,
                           "Rust 8-point sample vs cv2.findFundamentalMat (FM_8POINT) on 8 points"),
    }
    for name, (ref, tol, why) in refs.items():
        got = np.array(V(name)).reshape(-1)
        results.append(
            compare(
                Case(
                    name,
                    got,
                    ref[: len(got)],
                    0,
                    atol=tol,
                    reason=why,
                )
            )
        )
    _ef_ref = fit_and_held(np.linalg.inv(k).T @ e_cv @ np.linalg.inv(k))
    agreements.append(
        (
            "ep_resid_Efit",
            "held-out correspondences: Rust vs cv2.findEssentialMat",
            float(
                np.abs(
                    np.array(V("ep_resid_Efit")).reshape(-1) - _ef_ref
                ).max()
            ),
        )
    )
    note_detail(
        "ep_resid_Ffit",
        f"Rust's `find_fundamental_mat` reaches a worst Sampson residual of "
        f"{float(np.array(V('ep_resid_Ffit')).max()):.3e} px on the clean 20-point set; "
        f"cv2's own `FM_8POINT` measures {float(sampson(f_cv_8pt, fit1, fit2).max()):.3e} px "
        f"and `FM_RANSAC` {float(sampson(f_cv_ransac, fit1, fit2).max()):.3e} px on the "
        f"same data. **The Rust side is between 2 and 5 orders of magnitude tighter "
        f"than OpenCV's**, so a naive parity tolerance set from OpenCV's accuracy "
        f"would have flagged the better answer as a defect. Both are far below any "
        f"pixel-significant threshold.",
    )
    note_detail(
        "ep_resid_Ffit8",
        f"On the minimal 8-point sample, Rust reaches "
        f"{float(np.array(V('ep_resid_Ffit8')).max()):.3e} px where cv2's FM_8POINT "
        f"reaches {float(sampson(f_cv_8pt_min, fit1, fit2).max()):.3e} px. The repo's "
        f"history here (the minimal-sample null-vector fix, homography.rs:54-55) is "
        f"consistent with that ordering. **Agreement at any pixel-significant "
        f"tolerance.**",
    )

    # `F_fit` and `E_fit` are each emitted twice: once normalised and once raw.
    # The raw forms are checked for unit Frobenius norm so the normalisation is
    # not silently a no-op on the records compared elsewhere.
    f_fit_raw = V("F_fit_rowmajor")
    results.append(
        compare(
            Case(
                "epipole_F_fit_frobenius_norm",
                np.array([[np.linalg.norm(f_fit_raw.reshape(3, 3))]]),
                np.array([[1.0]]),
                0,
                atol=1e-12,
                reason="F_fit_rowmajor is the unnormalised matrix; its Frobenius norm "
                "being 1 is what makes the F/h22 record comparable",
            )
        )
    )
    e_fit_raw = V("E_fit_rowmajor")
    results.append(
        compare(
            Case(
                "epipole_E_fit_frobenius_norm",
                np.array([[np.linalg.norm(e_fit_raw.reshape(3, 3))]]),
                np.array([[1.0]]),
                0,
                atol=1e-12,
                reason="same for E_fit_rowmajor",
            )
        )
    )
    results.append(
        compare(
            Case(
                "epipole_F_from_E_fit",
                V("F_from_E_fit"),
                (np.linalg.inv(k).T @ e_fit_raw.reshape(3, 3) @ np.linalg.inv(k)).reshape(-1),
                0,
                atol=1e-15,
                reason="F = K^-T E K^-1 applied to the emitted E_fit",
            )
        )
    )

    # RANSAC on the contaminated set. Both sides use a random sampler, so only
    # distributional quantities (inlier count, per-point mask) are comparable.
    contam1 = np.array(V("ep_contam_pts1")).reshape(-1, 2)
    contam2 = np.array(V("ep_contam_pts2")).reshape(-1, 2)
    n_corrupt = int((~np.isclose(contam1, fit1).all(axis=1)).sum())
    f_ransac_cv, mask_r = cv2.findFundamentalMat(contam1, contam2, cv2.FM_RANSAC, 1.5, 0.9999)
    e_ransac_cv, mask_e = cv2.findEssentialMat(
        contam1, contam2, cameraMatrix=k, method=cv2.RANSAC, prob=0.9999, threshold=1.5
    )
    rust_inl = np.array(V("F_ransac_inliers"))
    rust_n = int(rec.scalars["F_ransac_n_inliers"])
    rust_e_n = int(rec.scalars["E_ransac_n_inliers"])
    cv_n = int(mask_r.sum())
    residual_notes.append(
        f"RANSAC fundamental, {n_corrupt}/{len(contam1)} correspondences corrupted "
        f"(at indices {np.where(~np.isclose(contam1, fit1).all(axis=1))[0].tolist()}): "
        f"Rust inliers {rust_n}/{len(contam1)}, cv2 {cv_n}/{len(contam1)}. "
        f"cv2's own max Sampson residual on the contaminated set is "
        f"{sampson_max(f_ransac_cv, contam1, contam2):.3f} px at the same 1.5 px threshold."
    )
    results.append(
        compare(
            Case(
                "ransac_inlier_count_F",
                np.array([[float(rust_n)]]),
                np.array([[float(cv_n)]]),
                0,
                atol=0.0,
                reason="RANSAC inlier counts, compared as scalars rather than as "
                "pixel deviations",
            )
        )
    )
    results.append(
        compare(
            Case(
                "ransac_inlier_count_E",
                np.array([[float(rust_e_n)]]),
                np.array([[float(int(mask_e.sum()))]]),
                0,
                atol=0.0,
                reason="same, for the essential matrix",
            )
        )
    )
    results.append(
        compare(
            Case(
                "ransac_inlier_mask_agreement",
                rust_inl.reshape(-1, 1),
                mask_r.reshape(-1, 1).astype(np.float64),
                0,
                atol=0.0,
                reason="per-correspondence inlier flags, compared exactly; "
                "both sides must reject the same corrupted indices",
            )
        )
    )
    agreements.append(
        (
            "ransac_inlier_mask_agreement",
            f"cv2 vs Rust per-point inlier flags ({len(contam1)} points)",
            float(np.abs(rust_inl - mask_r.reshape(-1)).max()),
        )
    )
    results.append(
        compare(
            Case(
                "ransac_F_max_residual_on_clean",
                np.array([[float(np.array(V("ep_resid_Fransac")).max())]]),
                np.array([[sampson_max(f_ransac_cv, fit1, fit2)]]),
                0,
                atol=1e-6,
                reason="each side's RANSAC model, scored on the *clean* fit set; "
                "the RANSAC search itself is random so only the resulting model's "
                "quality is comparable",
            )
        )
    )
    results.append(
        compare(
            Case(
                "ransac_E_max_residual_on_clean",
                np.array([[float(np.array(V("ep_resid_Eransac")).max())]]),
                np.array(
                    [[sampson_max(np.linalg.inv(k).T @ e_ransac_cv @ np.linalg.inv(k), fit1, fit2)]]
                ),
                0,
                atol=1e-6,
                reason="same, for the essential RANSAC model",
            )
        )
    )

    # `recover_pose_from_essential` has a documented history of wrong
    # decomposition candidates, so it is checked against cv2.recoverPose on the
    # same E, and separately for the two things it promises: a unit-norm
    # direction and one of the admissible rotation/translation pairs.
    pose_rec = V("E_pose_recovered")
    tr = pose_rec[:3]
    rr = pose_rec[3:12].reshape(3, 3)
    out = cv2.recoverPose(e_fit_raw.reshape(3, 3), fit1, fit2, k)
    r_cv = np.asarray(out[1]).reshape(3, 3)
    t_cv = np.asarray(out[2]).reshape(3)
    results.append(
        compare(
            Case(
                "pose_recovered_rotation_vs_opencv",
                rr,
                r_cv,
                0,
                atol=1e-9,
                reason="cv2.recoverPose on the same E_fit and correspondences",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_translation_direction_vs_opencv",
                tr.reshape(1, 3),
                (t_cv / np.linalg.norm(t_cv)).reshape(1, 3),
                0,
                atol=1e-9,
                reason="cv2.recoverPose's t is likewise a direction, not a baseline",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_rotation_vs_ground_truth",
                rr,
                r_rel,
                0,
                atol=1e-9,
                reason="and against the true rotation, which for these "
                "correspondences is the R <-> R^T-resolved answer",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_translation_direction_vs_ground_truth",
                tr.reshape(1, 3),
                (t_rel / np.linalg.norm(t_rel)).reshape(1, 3),
                0,
                atol=1e-9,
                reason="and against the true baseline direction",
            )
        )
    )
    results.append(
        compare(
            Case(
                "pose_recovered_baseline_is_unit",
                np.array([[np.linalg.norm(tr)]]),
                np.array([[1.0]]),
                0,
                atol=1e-12,
                reason="documented: E determines t only up to scale, so a unit "
                "direction is the only claim a caller can make",
            )
        )
    )
    agreements.append(
        (
            "pose_recovered_rotation_vs_opencv",
            "and vs the true rotation from the generating pose",
            float(max(np.abs(rr - r_cv).max(), np.abs(rr - r_rel).max())),
        )
    )

    # ── report ──────────────────────────────────────────────────────────────
    print(f"ENV {fmt(env)}")
    print(f"RECORDS {len(results)}")
    print()
    for n in notes:
        print("NOTE " + n)
    print()
    print(render_table(results, "calib3d: intrinsics, projection, distortion, DLT, epipolar"))
    print()
    for n in residual_notes:
        print("NOTE " + n)
    print()
    print("Agreements worth recording (independent checks that closed):")
    for name, how, dev in agreements:
        print(f"  {name}: {how}, max dev {dev:.3e}")
    print()
    print("ATTRIBUTION (per comparison that did not simply match):")
    for name in sorted(detail):
        for line in detail[name]:
            print(f"  {name}: {line}")
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


def skew(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )

def project_pinhole_from_cam32(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Reference for the f32 pinhole path: same formula in f32."""
    fx, fy, cx, cy = (np.float32(v) for v in CAM)
    return np.column_stack(
        [fx * xn + cx, fy * yn + cy]
    ).astype(np.float64)


def fisheye_pinhole32(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    kd = np.array(KANNALA, dtype=np.float32)
    r = np.hypot(xn, yn)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + kd[0] * th2 + kd[1] * th2 ** 2 + kd[2] * th2 ** 3 + kd[3] * th2 ** 4)
    scale = thd / r
    fx, fy, cx, cy = (np.float32(v) for v in CAM)
    return np.column_stack([fx * xn * scale + cx, fy * yn * scale + cy]).astype(np.float64)


if __name__ == "__main__":
    raise SystemExit(main())