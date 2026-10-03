"""Parity: cv-imgproc resampling & geometry vs OpenCV 4.13.

Reference side only. Regenerates the identical deterministic inputs and the
identical affine matrix / displacement field, recomputes each quantity with
cv2, and compares against the Rust record file.

Run:  python3 parity/parity_geometry.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    Case,
    affine_rotate_translate,
    compare,
    diverging,
    environment,
    fmt,
    geometry_inputs,
    load,
    remap_field,
    render_table,
    run_rust,
)

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

BORDER = {"c0": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE}

# ── tolerances ──────────────────────────────────────────────────────────────
#
# All four cases below are u8-in / u8-out on both sides, so both quantise once
# to the same 8-bit grid. What differs is the *sample value* before that step.
#
# * INTER_NEAREST: the sampled integer coordinate is exact on both sides, so
#   the result must be bit-identical. Any deviation is a coordinate-mapping
#   bug. Tolerance 0.
ATOL_NEAREST = 0.0
#
# * INTER_LINEAR / INTER_CUBIC / INTER_LANCZOS4: both sides compute a real-valued
#   weighted sum from identical weights and identical taps, then round to
#   nearest. The only difference is f32 (Rust) vs f64 (OpenCV) accumulation,
#   bounded here by |v| * 1e-6 ~ 3e-4 counts - far below a tie. Tolerance 0.5
#   counts admits only a rounding-tie difference, which is the smallest
#   non-zero difference the u8 output grid can express.
ATOL_INTERP = 0.5


def u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0, 255).astype(np.uint8)


def main() -> int:
    path = run_rust("cv-imgproc", "parity_geometry")
    rec = load(path)
    env = environment()
    results = []
    notes = []

    inputs = geometry_inputs()
    for name, expected in inputs.items():
        got = rec.images[f"input_{name}"]
        if got.shape != expected.shape or not np.array_equal(got, expected):
            print(f"INPUT MISMATCH {name}", file=sys.stderr)
            return 2
    notes.append(
        "Both geometry inputs (a smooth 2x1-cycle sinusoid, and an 8-px "
        "checkerboard chosen because it aliases maximally under downscale) "
        "are byte-identical between the two sides."
    )

    # ── resize ──────────────────────────────────────────────────────────────
    # cv2.resize INTER_NEAREST uses floor(x * (sw/dw)), which is what
    # resize_nearest implements. The linear/cubic/lanczos cases are the
    # interesting ones: OpenCV's coordinate mapping is
    #   fx = (dx + 0.5) * scale - 0.5   (half-pixel centres)
    # which is a different convention from the align-corners mapping used by
    # the Rust bilinear HAL path.
    for iname, img in inputs.items():
        src = u8(img)
        for tw, th in [(24, 18), (96, 72), (13, 40)]:
            for mname, mcode in INTERP.items():
                case = f"resize_{iname}_{mname}_{tw}x{th}"
                tol = ATOL_NEAREST if mname == "nearest" else ATOL_INTERP
                results.append(
                    compare(
                        Case(
                            case,
                            rec.images[case],
                            cv2.resize(
                                src, (tw, th), interpolation=mcode
                            ).astype(np.float64),
                            rec.meta[case],
                            atol=tol,
                            reason="u8 in / u8 out, single shared quantisation step",
                        )
                    )
                )

    # ── warpAffine ──────────────────────────────────────────────────────────
    # The matrix is read back out of the record file: the Rust side receives it
    # as f32, so the reference must use exactly those f32-rounded coefficients,
    # not the f64 ones, or the comparison would measure input drift.
    m = np.array(
        [[rec.params["warpM00"][0], rec.params["warpM01"][0], rec.params["warpM02"][0]],
         [rec.params["warpM10"][0], rec.params["warpM11"][0], rec.params["warpM12"][0]]]
    )
    notes.append(
        "The 20-degree rotation + translation matrix is echoed by the Rust "
        f"side as f32 and the reference uses those exact coefficients "
        f"({m.tolist()}), so a deviation cannot be matrix-rounding drift."
    )
    for iname, img in inputs.items():
        src = u8(img)
        for bname, bcode in BORDER.items():
            case = f"warpaffine_{iname}_linear_{bname}"
            results.append(
                compare(
                    Case(
                        case,
                        rec.images[case],
                        cv2.warpAffine(
                            src, m, (img.shape[1], img.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=bcode,
                        ).astype(np.float64),
                        rec.meta[case],
                        atol=ATOL_INTERP,
                        reason="bilinear, identical coefficients and border mode",
                    )
                )
            )
        case = f"warpaffine_{iname}_nearest_c0"
        results.append(
            compare(
                Case(
                    case,
                    rec.images[case],
                    cv2.warpAffine(
                        src, m, (img.shape[1], img.shape[0]),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                    ).astype(np.float64),
                    rec.meta[case],
                    atol=ATOL_NEAREST,
                    reason="nearest isolates the coordinate transform from the kernel",
                )
            )
        )

    # ── remap ───────────────────────────────────────────────────────────────
    mx, my = remap_field()
    notes.append(
        "The displacement field is regenerated from the same closed form on "
        "both sides: map_x = x + 3 + 0.5*y^2/H, map_y = y - 2 + 0.5*x^2/W, "
        "cast to f32 by the Rust side and mirrored exactly here."
    )
    mx32 = mx.astype(np.float32)
    my32 = my.astype(np.float32)
    for iname, img in inputs.items():
        src = u8(img)
        for mname, mcode in INTERP.items():
            case = f"remap_{iname}_{mname}_c0"
            tol = ATOL_NEAREST if mname == "nearest" else ATOL_INTERP
            results.append(
                compare(
                    Case(
                        case,
                        rec.images[case],
                        cv2.remap(
                            src, mx32, my32, (img.shape[1], img.shape[0]),
                            mcode, borderMode=cv2.BORDER_CONSTANT,
                        ).astype(np.float64),
                        rec.meta[case],
                        atol=tol,
                        reason="identical map arrays, single shared quantisation step",
                    )
                )
            )

    print(f"ENV {fmt(env)}")
    print(f"RECORDS {len(results)}")
    print()
    for n in notes:
        print("NOTE " + n)
    print()
    print(render_table(results, "Resampling & geometry: cv-imgproc vs OpenCV"))
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


if __name__ == "__main__":
    raise SystemExit(main())