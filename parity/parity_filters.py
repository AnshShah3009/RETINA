"""Parity: cv-imgproc filters vs OpenCV 4.13.

Reference side only. Regenerates the identical deterministic inputs, recomputes
each quantity with cv2, and compares against the Rust record file line by line.

Run:  python3 parity/parity_filters.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    Case,
    compare,
    diverging,
    environment,
    filter_inputs,
    fmt,
    load,
    morph_kernel,
    render_table,
    separable_f64,
    run_rust,
)

# ── tolerances, each justified ───────────────────────────────────────────────
#
# Both sides emit u8. A normalised smoother on the same kernel with the same
# border mode has the same exact arithmetic (products of u8 x f32 weights) in
# the same order for gaussian (separable, f32 accumulation), so the only
# divergence is float rounding plus any round-to-u8 tie. That is bounded by
# half an ulp of f32 relative to the accumulated magnitude: < 1e-3 counts for
# these ranges. A 0.5-count tolerance is therefore comfortably above the
# legitimate noise floor and far below one quantisation step - it can only be
# tripped by a real difference, never by rounding.
ATOL_IMAGE = 0.5

# Morphology is min/max over integers: exact, no floating point anywhere. The
# tolerance must therefore be 0, and any non-zero deviation is a defect rather
# than a rounding artefact.
ATOL_MORPH = 0.0

# Sobel: the Rust result is u8 (clamped), so a like-for-like comparison is
# against the reference rounded to nearest u8. OpenCV's CV_64F Sobel is exact
# integer arithmetic in its own path but our value differs by <= 1 count from
# the mathematical convolution because the intermediate horizontal pass is
# rounded to u8. Hence 1.0.
ATOL_SOBEL = 1.0


def u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def gaussian_ref(img: np.ndarray, sigma: float, border: int) -> np.ndarray:
    """OpenCV GaussianBlur with the kernel size OpenCV itself would derive.

    cv2.GaussianBlur(ksize=(0,0), sigma) uses ksize = 2*cvRound(3*sigma)+1.
    The Rust path uses (ceil(6*sigma))|1. These coincide for every sigma tested
    (0.75, 1.5, 2.0), which is exactly why only those sigmas are compared:
    otherwise the kernels have different supports and the measurement would be
    of the window, not of the filter.
    """
    ksize = (0, 0)
    return cv2.GaussianBlur(u8(img), ksize, sigmaX=sigma, borderType=border)


def sobel_ref(img: np.ndarray, dx: int, dy: int, ksize: int) -> tuple[np.ndarray, float]:
    """CV_64F Sobel plus the reference's own rounded-to-u8 view.

    Returns (rounded, max_abs_reference) so the harness can report how much of
    the difference is attributable to the sign/quantisation loss in the Rust
    u8 output rather than to the convolution itself.
    """
    ref = cv2.Sobel(
        u8(img), cv2.CV_64F, dx=dx, dy=dy, ksize=ksize, scale=1.0, delta=0.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    rounded = np.clip(np.round(ref), 0, 255)
    return rounded, float(np.abs(ref).max())


def main() -> int:
    path = run_rust("cv-imgproc", "parity_filters")
    rec = load(path)
    env = environment()

    results = []
    notes = []

    # ── input identity check (the contract that makes the rest meaningful) ──
    inputs = filter_inputs()
    for name, expected in inputs.items():
        got = rec.images[f"input_{name}"]
        if got.shape != expected.shape or not np.array_equal(got, expected):
            print(f"INPUT MISMATCH for {name}:\n{got}\n{expected}", file=sys.stderr)
            return 2
    notes.append(
        "All 4 test inputs (constant, impulse, unit-slope ramp, 3.0x2.0-cycle "
        "sinusoid) were regenerated independently on the Python side and are "
        "**byte-identical** to the Rust inputs. Every deviation below is "
        "therefore attributable to the filter, not to the input."
    )

    # ── gaussian blur ───────────────────────────────────────────────────────
    border_map = {
        "reflect101": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
    }
    for iname in inputs:
        for sigma in (0.75, 1.5, 2.0):
            for bname, bmode in border_map.items():
                case = f"gauss_{iname}_s{sigma}_{bname}"
                radius = rec.meta[case]
                results.append(compare(Case(
                        case,
                        rec.images[case],
                        gaussian_ref(inputs[iname], sigma, bmode).astype(np.float64),
                        radius,
                        atol=ATOL_IMAGE,
                        reason="same separable f32 kernel, same border mode",
                    )
                ))

    # ── erode / dilate ──────────────────────────────────────────────────────
    # Two kernel cases per input:
    #   *kernel_identical: cv2 gets exactly the Rust support (rect5, and the
    #    Rust ellipse convention) - isolates the algorithm.
    #   *kernel_opencv:    cv2 gets its own cv2.getStructuringElement - asks
    #    the question a replacement actually faces.
    opencv_ellipse5 = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) > 0).astype(
        np.float64
    )
    rust_ellipse5 = morph_kernel("ellipse", 5)
    notes.append(
        "The 5x5 **ellipse** structuring element differs between the two sides "
        f"and that is a genuine definitional difference, not noise: the Rust "
        f"`create_morph_kernel(MorphShape::Ellipse, 5, 5)` support has "
        f"{int(rust_ellipse5.sum())} pixels, `cv2.getStructuringElement("
        f"MORPH_ELLIPSE, (5,5))` has {int(opencv_ellipse5.sum())}. OpenCV uses a "
        "radius-0.5 convention; the Rust code uses the inclusive radius with "
        "cx = k/2. Both cases are reported."
    )
    for iname, img in inputs.items():
        for ktag, kmask, kocv in [
            ("rect5", morph_kernel("rect", 5), morph_kernel("rect", 5)),
            ("ellipse5", rust_ellipse5, opencv_ellipse5),
        ]:
            src = u8(img)
            results.append(compare(Case(
                    f"dilate_{iname}_{ktag}",
                    rec.images[f"dilate_{iname}_{ktag}"],
                    cv2.dilate(src, kocv.astype(np.uint8), iterations=1,
                               borderType=cv2.BORDER_REPLICATE).astype(np.float64),
                    rec.meta[f"dilate_{iname}_{ktag}"],
                    atol=ATOL_MORPH,
                    reason="min/max over integers; exact comparison",
                )
            ))
            results.append(compare(Case(
                    f"erode_{iname}_{ktag}",
                    rec.images[f"erode_{iname}_{ktag}"],
                    cv2.erode(src, kocv.astype(np.uint8), iterations=1,
                              borderType=cv2.BORDER_REPLICATE).astype(np.float64),
                    rec.meta[f"erode_{iname}_{ktag}"],
                    atol=ATOL_MORPH,
                    reason="min/max over integers; exact comparison",
                )
            ))
            # kernel_identical variants, using the Rust support on both sides
            if not np.array_equal(kmask, kocv):
                for op, cvop in [("dilate", cv2.dilate), ("erode", cv2.erode)]:
                    results.append(compare(Case(
                            f"{op}_{iname}_{ktag}_kernelidentical",
                            rec.images[f"{op}_{iname}_{ktag}"],
                            cvop(src, kmask.astype(np.uint8), iterations=1,
                                 borderType=cv2.BORDER_REPLICATE).astype(np.float64),
                            2,
                            atol=ATOL_MORPH,
                            reason="same support on both sides; isolates the algorithm",
                        )
                    ))

    # ── Sobel ───────────────────────────────────────────────────────────────
    for iname, img in inputs.items():
        for dx, dy in [(1, 0), (0, 1), (1, 1)]:
            for ksize in (3, 5):
                case = f"sobel_{iname}_dx{dx}_dy{dy}_k{ksize}"
                ref, ref_absmax = sobel_ref(img, dx, dy, ksize)
                results.append(compare(Case(
                        case,
                        rec.images[case],
                        ref.astype(np.float64),
                        rec.meta[case],
                        atol=ATOL_SOBEL,
                        reason=(
                            f"reference CV_64F magnitude max={ref_absmax:.1f}; "
                            "Rust output is u8 so comparison is against round(ref)"
                        ),
                    )
                ))

    print(f"ENV {fmt(env)}")
    print(f"RECORDS {len(results)}")
    print()
    for n in notes:
        print("NOTE " + n)
    print()

    # ── attribution probe for the gaussian off-by-one ───────────────────────
    # Rust quantises the separable f32 convolution with `as u8`, i.e. it
    # *truncates*; OpenCV's `filter2D`/`GaussianBlur` round to nearest. The
    # probe below applies the library's own kernel coefficients in f64 with no
    # quantisation at all. If that agrees with the OpenCV reference to within
    # f64 rounding, then the entire discrepancy is the quantisation step and
    # both filters are the same filter.
    print("PROBE gaussian kernel identity + quantisation attribution")
    for sigma in (0.75, 1.5, 2.0):
        kkey = f"s{sigma}"
        k = np.array(rec.rows_f32[f"kernel1d_{kkey}"], dtype=np.float64)
        sumk = k.sum()
        # The library also prints its own f32 accumulator sum of the same
        # kernel; agreement between the two confirms the kernel really is
        # normalised to 1 (a DC gain of exactly 1 is what makes a blurred
        # constant reproduce that constant).
        reported = [r for r in rec.rows if r[0] == kkey and len(r) == 3]
        lib_sum = float(reported[0][1]) if reported else float("nan")
        agree = abs(lib_sum - sumk) <= 1e-5
        line = (
            f"  sigma={sigma}: kernel size={k.size}, sum={sumk:.17e} "
            f"(library's own f32 sum {lib_sum:.17e}, normalised={'yes' if agree else 'NO'})"
        )
        for iname, img in inputs.items():
            exact = separable_f64(img.astype(np.float64), k, border="reflect101")
            rounded = np.clip(np.round(exact), 0, 255)
            ref = gaussian_ref(img, sigma, cv2.BORDER_REFLECT_101).astype(np.float64)
            d_rounded = float(np.abs(rounded - ref).max())
            line += f"\n    {iname}: max|rust_kernel_in_f64_rounded - cv2| = {d_rounded:g}"
        print(line)
    print()
    print(render_table(results, "Filters: cv-imgproc vs OpenCV"))
    print()

    from harness import diverging

    bad = diverging(results)
    print(f"DIVERGING {len(bad)} of {len(results)}")
    for r in bad:
        y, x = r.argmax
        print(
            f"  {r.name}: max={r.max_abs:.6g} at (x={x}, y={y}) "
            f"rust={r.rust_at_max:.6g} ref={r.ref_at_max:.6g} "
            f"over_tol={r.n_over}/{r.n} ({r.frac_over:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
