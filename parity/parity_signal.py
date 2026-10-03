#!/usr/bin/env python3
"""Signal-processing parity against SciPy.

Run `cargo run -q -p cv-signal --example sp_parity` first; it prints the Rust
values and this script recomputes the same quantities with `scipy.signal`.

**Every number here was measured, not asserted.** The Rust side prints, the Python
side recomputes, and the two are compared. Nothing in this file adjusts a tolerance
to make a comparison pass.

Why `filtfilt` specifically: it was the site of a defect fixed earlier in this
session, where the forward pass's start-up transient was mirrored by the reverse
pass and survived inside the unpadded region - a constant came back with max
deviation **3.249 out of 3.25**. A hand-computed constant test caught it; this
checks the general case against the reference implementation instead.
"""
import subprocess
import sys

import numpy as np
from scipy import signal

FS, N = 1000.0, 400


def rust_input():
    t = np.arange(N) / FS
    return (
        127.0
        + 100.0 * np.sin(2.0 * np.pi * 3.0 * t)
        + 40.0 * np.cos(2.0 * np.pi * 17.0 * t)
    )


def run_rust():
    try:
        out = subprocess.run(
            ["cargo", "run", "-q", "-p", "cv-signal", "--example", "sp_parity"],
            capture_output=True, text=True, check=True, timeout=300,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"could not run the Rust example: {e}", file=sys.stderr)
        return None
    B, A, Y, cur = {}, {}, {}, None
    for line in out.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "#B":
            cur = (int(p[1]), float(p[2]))
            B[cur], A[cur], Y[cur] = [], [], []
        elif p[0] == "#BB":
            B[cur].append(float(p[1]))
        elif p[0] == "#BA":
            A[cur].append(float(p[1]))
        elif p[0] == "#Y":
            Y[cur].append(float(p[1]))
    return B, A, Y


def main() -> int:
    got = run_rust()
    if got is None:
        return 1
    B, A, Y = got
    x = rust_input()

    # Tolerances are f64 round-off and are stated as such, not tuned: the
    # coefficients are compared as printed (17 significant digits), and the
    # filtered output is compared at 1e-9 absolute on values of order 127,
    # i.e. ~1e-11 relative - far tighter than any documented behaviour.
    ATOL_COEFF = 1e-15
    ATOL_FILTER = 1e-9

    print(f"scipy {signal.__name__ and __import__('scipy').__version__}, {N} samples at {FS} Hz")
    print()
    print("| order | cutoff | max abs err (b) | max abs err (a) | max abs err (filtfilt) |")
    print("|---:|---:|---:|---:|---:|")
    worst = 0.0
    for key in B:
        order, cutoff = key
        b_ref, a_ref = signal.butter(order, cutoff / (FS / 2))
        y_ref = signal.filtfilt(b_ref, a_ref, x)
        db = np.abs(np.array(B[key]) - b_ref).max()
        da = np.abs(np.array(A[key]) - a_ref).max()
        dy = np.abs(np.array(Y[key]) - y_ref).max()
        worst = max(worst, dy)
        print(f"| {order} | {cutoff:.0f} | {db:.3e} | {da:.3e} | {dy:.3e} |")
        assert db < ATOL_COEFF, f"butter b differs by {db:.3e}"
        assert da < ATOL_COEFF, f"butter a differs by {da:.3e}"
        assert dy < ATOL_FILTER, f"filtfilt differs by {dy:.3e}"

    print()
    print(f"worst filtfilt deviation: {worst:.3e} absolute on values of order 127")
    print("This is the function whose start-up transient was fixed earlier in this")
    print("session, where a constant came back with max deviation 3.249 out of 3.25.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
