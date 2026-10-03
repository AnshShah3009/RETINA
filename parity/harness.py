"""Shared machinery for the RETINA numerical-parity harness.

The Rust side (`crates/*/examples/parity_*.rs`) writes a text file of records.
This module parses it, and provides the comparison/reporting primitives. It
knows nothing about how Rust computed the numbers - only the record protocol.

Record protocol
---------------
    #IM <name> <width> <height> <v0> <v1> ...   image, row-major, f64
    #W  <name> <width> <height> <v0> ...         warped/resampled image
    #M  <name> <radius>                          case metadata
    #V  <name> <value>                            scalar
    #B  <name> <d> <sigma_color> <sigma_space>   bilateral params
    #C  <name> <clip_limit> <tiles_x> <tiles_y>  CLAHE params
    #F  <name> <value>                            f32 parameter (geometry matrices)
    <name> <x> <value>                           cv-math scalars, one line each

The single sharpest contract in the harness is input identity: the reference
side regenerates every input from the same closed-form definition the Rust
example uses and asserts byte-identity. A drifting input generator then fails
loudly instead of yielding a meaningless comparison.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("PARITY_SCRATCH", "/tmp/retina-parity")


# ── record file parsing ─────────────────────────────────────────────────────


@dataclass
class Record:
    images: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (h, w) u8-as-f64
    meta: dict[str, int] = field(default_factory=dict)  # name -> border radius
    scalars: dict[str, float] = field(default_factory=dict)  # name -> value
    params: dict[str, tuple[float, ...]] = field(default_factory=dict)
    rows: list[tuple[str, ...]] = field(default_factory=list)  # cv-math triples
    rows_f32: dict[str, list[float]] = field(default_factory=dict)  # name -> f32 row


def load(path: str) -> Record:
    rec = Record()
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#IM ") or line.startswith("#W "):
                _, name, w, h, *vals = line.split()
                rec.images[name] = np.array(vals, dtype=np.float64).reshape(int(h), int(w))
            elif line.startswith("#M "):
                _, name, radius = line.split()
                rec.meta[name] = int(radius)
            elif line.startswith("#V "):
                _, name, value = line.split()
                rec.scalars[name] = float(value)
            elif line.startswith("#B "):
                _, name, d, sc, ss = line.split()
                rec.params[name] = (float(d), float(sc), float(ss))
            elif line.startswith("#C "):
                _, name, clip, tx, ty = line.split()
                rec.params[name] = (float(clip), float(tx), float(ty))
            elif line.startswith("#F "):
                _, name, value = line.split()
                rec.params[name] = (float(value),)
            elif line.startswith("#FR "):
                _, name, _n, *vals = line.split()
                rec.rows_f32[name] = [float(v) for v in vals]
            elif line.startswith("#FK "):
                rec.rows.append(tuple(line.split()[1:]))
            elif line.startswith("#"):
                raise ValueError(f"unrecognised record: {line!r}")
            else:
                rec.rows.append(tuple(line.split()))
    return rec


# ── deterministic input generators (mirrored from the Rust examples) ─────────

W_F, H_F = 64, 48
TAU = 2.0 * np.pi


def _put(img: np.ndarray, x: int, y: int, v: float) -> None:
    img[y, x] = int(np.clip(round(v), 0, 255))


def filter_inputs() -> dict[str, np.ndarray]:
    """Inputs for crates/imgproc/examples/parity_filters.rs."""
    const = np.full((H_F, W_F), 128.0)
    impulse = np.zeros((H_F, W_F))
    impulse[H_F // 2, W_F // 2] = 255.0
    ramp = np.add.outer(np.arange(H_F, dtype=np.float64), np.arange(W_F, dtype=np.float64))
    yy, xx = np.mgrid[0:H_F, 0:W_F]
    sinusoid = 127.0 + 100.0 * np.sin(TAU * (3.0 * xx / W_F + 2.0 * yy / H_F))
    return {
        "const": const,
        "impulse": impulse,
        "ramp": ramp,
        "sinusoid": np.round(sinusoid),
    }


def morph_kernel(shape: str, k: int) -> np.ndarray:
    """Mirror of `cv_imgproc::create_morph_kernel`.

    The rectangle is the full k x k window. The ellipse uses cx = k/2 with the
    *inclusive* radius (dx^2/rx^2 + dy^2/ry^2 <= 1) evaluated on the same grid -
    which is exactly what the Rust code does. OpenCV's `getStructuringElement`
    uses a different (radius - 0.5) convention, so the harness compares both
    the same-kernel case and OpenCV's own kernel and reports the difference.
    """
    m = np.zeros((k, k), dtype=np.float64)
    cx = k // 2
    if shape == "rect":
        m[:, :] = 1.0
    elif shape == "ellipse":
        rx = k / 2.0
        ry = k / 2.0
        for y in range(k):
            for x in range(k):
                dx = float(x - cx)
                dy = float(y - cx)
                if (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0:
                    m[y, x] = 1.0
    else:
        raise ValueError(shape)
    return m


W_G, H_G = 48, 36


def geometry_inputs() -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:H_G, 0:W_G]
    smooth = np.round(127.0 + 100.0 * np.sin(TAU * (2.0 * xx / W_G + 1.0 * yy / H_G)))
    checker = np.where((xx + yy) % 2 == 0, 40.0, 200.0)
    return {"smooth": smooth.astype(np.float64), "checker": checker.astype(np.float64)}


def affine_rotate_translate(angle_deg: float, tx: float, ty: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    cx = (W_G - 1) / 2.0
    cy = (H_G - 1) / 2.0
    m = np.array(
        [
            [c, s, (1 - c) * cx + s * cy + tx],
            [-s, c, s * cx - (1 - c) * cy + ty],
        ],
        dtype=np.float64,
    )
    return m


def remap_field() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:H_G, 0:W_G].astype(np.float64)
    mx = xx + 3.0 + 0.5 * yy * yy / H_G
    my = yy - 2.0 + 0.5 * xx * xx / W_G
    return mx, my


W_P, H_P = 64, 64


def photo_inputs() -> dict[str, np.ndarray]:
    xx = np.arange(W_P, dtype=np.float64)
    yy = np.mgrid[0:H_P, 0:W_P][0].astype(np.float64)
    low = np.round(100.0 + 60.0 * xx / (W_P - 1.0))
    low = np.repeat(low[None, :], H_P, axis=0)
    fx = xx / (W_P - 1.0)
    base = np.where(fx < 0.5, 30.0, 200.0)
    wobble = 20.0 * np.sin(TAU * (2.0 * yy / H_P))
    bimodal = np.round(base[None, :] + wobble)
    yy2, xx2 = np.mgrid[0:H_P, 0:W_P]
    bil = np.round(127.0 + 100.0 * np.sin(TAU * (3.0 * xx2 / W_P + 2.0 * yy2 / H_P)))
    return {"low_contrast": low, "bimodal": bimodal, "bilateral": bil}


# ── comparison primitives ───────────────────────────────────────────────────


@dataclass
class Case:
    name: str
    rust: np.ndarray
    ref: np.ndarray
    radius: int
    note: str = ""
    #: absolute tolerance in output units (u8 counts for images, 1.0 for
    #: unit-range scalars)
    atol: float = 0.0
    reason: str = ""


@dataclass
class Result:
    name: str
    n: int
    max_abs: float
    max_abs_interior: float
    n_over: int
    frac_over: float
    note: str
    reason: str
    tol: float
    argmax: tuple[int, int] | None = None
    rust_at_max: float = 0.0
    ref_at_max: float = 0.0

    @property
    def pass_full(self) -> bool:
        return self.max_abs <= self.tol

    @property
    def pass_interior(self) -> bool:
        return self.max_abs_interior <= self.tol


def compare(case: Case) -> Result:
    rust = np.asarray(case.rust, dtype=np.float64)
    ref = np.asarray(case.ref, dtype=np.float64)
    if rust.shape != ref.shape:
        return Result(
            case.name, 0, float("inf"), float("inf"), 0, 1.0,
            case.note, f"shape mismatch {rust.shape} vs {ref.shape}", case.atol,
        )
    d = np.abs(rust - ref)
    over = d > case.atol
    r = case.radius
    if r > 0 and rust.shape[0] > 2 * r and rust.shape[1] > 2 * r:
        interior = d[r:-r, r:-r]
    else:
        interior = d
    flat = int(np.argmax(d))
    y, x = divmod(flat, d.shape[1])
    res = Result(
        case.name,
        int(d.size),
        float(d.max()),
        float(interior.max()),
        int(over.sum()),
        float(over.mean()),
        case.note,
        case.reason,
        case.atol,
        (int(y), int(x)),
        float(rust[y, x]),
        float(ref[y, x]),
    )
    return res


def scalar_dev(name: str, rust: float, ref: float, atol: float, note: str = "", reason: str = ""):
    """Compare two scalars as a one-pixel Case so reporting is uniform."""
    c = Case(name, np.array([[rust]]), np.array([[ref]]), 0, note, atol, reason)
    return compare(c)


def separable_f64(img: np.ndarray, k: np.ndarray, border: str = "reflect101") -> np.ndarray:
    """Separable convolution of `img` with 1-D kernel `k`, in f64, unquantised.

    Used only as an *attribution probe*: it applies the kernel the library
    actually built to the reference input, so any residual difference against
    OpenCV cannot be the kernel. Border folding mirrors cv-imgproc's
    `map_coord` (Reflect101 / Replicate / Constant(0)).
    """
    r = k.size // 2

    def fold_indices(idx: np.ndarray, n: int, mode: str) -> np.ndarray:
        """Map out-of-range coordinates into 0..n-1 the way map_coord does."""
        if mode == "reflect101":
            # period 2n-2 with `period - m`, i.e. index -1 -> 1 (the border
            # pixel is NOT duplicated). Matches map_coord's Reflect101 branch.
            if n == 1:
                return np.zeros_like(idx)
            period = 2 * n - 2
            m = np.mod(idx, period)
            return np.where(m >= n, period - m, m)
        if mode == "replicate":
            return np.clip(idx, 0, n - 1)
        return np.where((idx < 0) | (idx >= n), 0, idx)

    h, w = img.shape

    # Horizontal pass: fold the source columns, then convolve.
    pad = np.pad(img, ((0, 0), (r, r)), mode="constant")
    cols = pad[:, fold_indices(np.arange(-r, w + r), w, border) + r]
    horiz = np.zeros_like(img, dtype=np.float64)
    for i, kv in enumerate(k):
        horiz += kv * cols[:, i : i + w]

    # Vertical pass: fold the horizontal result's rows, then convolve. The
    # intermediate is NOT quantised, matching the library's f32 scratch buffer.
    pad2 = np.pad(horiz, ((r, r), (0, 0)), mode="constant")
    rows = pad2[fold_indices(np.arange(-r, h + r), h, border) + r, :]
    vert = np.zeros_like(img, dtype=np.float64)
    for i, kv in enumerate(k):
        vert += kv * rows[i : i + h, :]
    return vert


# ── environment capture ─────────────────────────────────────────────────────


def environment() -> dict[str, str]:
    import cv2
    import scipy

    env = {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
        "opencv_build_threads": str(cv2.getNumThreads()),
    }
    try:
        out = subprocess.run(
            ["rustc", "-Vv"], capture_output=True, text=True, timeout=30
        ).stdout
        for line in out.splitlines():
            if line.startswith("release:"):
                env["rustc"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return env


def fmt(env: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in env.items())


# ── report rendering ────────────────────────────────────────────────────────


def render_table(results: list[Result], title: str) -> str:
    lines = [f"### {title}", ""]
    lines.append("| case | tol | max abs (full) | max abs (interior) | over tol | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        tol = r.tol
        if r.n == 0:
            verdict = "SHAPE MISMATCH"
        elif r.max_abs <= tol:
            verdict = "match"
        elif r.max_abs_interior <= tol:
            verdict = "border only"
        else:
            verdict = "**DEVIATES**"
        lines.append(
            f"| `{r.name}` | {tol:g} | {r.max_abs:.6g} | {r.max_abs_interior:.6g} | "
            f"{r.n_over}/{r.n} ({r.frac_over:.1%}) | {verdict} |"
        )
    return "\n".join(lines)


def diverging(results: list[Result]) -> list[Result]:
    return [
        r
        for r in results
        if r.n > 0 and r.max_abs > r.tol and r.max_abs_interior > r.tol
    ]


def run_rust(crate: str, example: str, profile: str = "release") -> str:
    """Build and run a parity example, returning the record file path."""
    os.makedirs(SCRATCH, exist_ok=True)
    out = os.path.join(SCRATCH, f"{example}.txt")
    build = subprocess.run(
        ["cargo", "build", "-p", crate, "--example", example, "--profile", profile],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr)
        raise SystemExit(f"build failed for {crate}/{example}")
    with open(out, "w") as fh:
        proc = subprocess.run(
            ["cargo", "run", "--quiet", "-p", crate, "--example", example, "--profile", profile],
            cwd=REPO_ROOT,
            stdout=fh,
            stderr=subprocess.PIPE,
            text=True,
        )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"run failed for {crate}/{example}")
    return out
