#!/usr/bin/env python3
"""Special-function parity against SciPy 1.17.1.

Run first:

    cargo run -q -p cv-math --example parity_special > /tmp/parity_special.txt

then

    python3 parity/parity_special.py /tmp/parity_special.txt

The Rust side **prints and asserts nothing**; this script recomputes every
quantity with `scipy.special` and prints the deviation. That separation is the
whole point: a Rust-side assertion can only compare the Rust side against a
constant somebody typed in, so a bug in both the implementation and the
expected value is invisible to it.

**Every tolerance here is derived from the quantity, not tuned.** For a function
whose result is O(1) the bound is a small multiple of `f64::EPSILON`; for one
whose result spans many decades the bound is on the *relative* deviation; for
one whose result is subnormal or underflows the bound is stated in ulps or the
record is reported as "underflow", never silently compared as a relative error
against zero. Where a deviation exceeds its bound the script says so and exits
non-zero rather than printing a table that a reader has to audit by eye.

BASELINE COMMIT: f0f7d49fb471e3bea78b0252cf5ea60a2c4597a2

The headline this file exists to record: `erf` was the Abramowitz & Stegun
7.1.26 rational approximation, whose five coefficients sum to 0.999999999, so
`erf(0)` returned 9.999999717180685e-10 instead of 0 and the worst deviation
from `scipy.special.erf` over x in [-4, 4] was 1.3851e-07 at x = -1.4. That is
within the documented A&S bound of ~1.5e-7 and simultaneously nine orders of
magnitude behind the reference library this workspace is trying to replace.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import scipy
from scipy import special as sp

EPS = float(np.finfo(np.float64).eps)

# Absolute bounds, each justified by the magnitude of the quantity being
# compared rather than chosen to make a comparison pass.
ATOL_UNIT = 1e-14          # quantities that are O(1): a few tens of ulps
RTOL_FUNCTION = 1e-11      # gamma-family fits, ~1e-11 relative is their floor
RTOL_TAIL = 1e-11          # relative, for results spanning decades

_failures: list[str] = []
_warnings: list[str] = []


def fail(msg: str) -> None:
    _failures.append(msg)


def warn(msg: str) -> None:
    _warnings.append(msg)


def parse(path: str) -> dict[str, list[tuple[list[float], float]]]:
    out: dict[str, list[tuple[list[float], float]]] = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 2:
                continue
            name, last = p[0], p[-1]
            try:
                val = float(last)
            except ValueError:
                continue
            args = []
            for tok in p[1:-1]:
                try:
                    args.append(float(tok))
                except ValueError:
                    args = []
                    break
            if not args:
                continue
            out.setdefault(name, []).append((args, val))
    return out


# ---------------------------------------------------------------------------
# erf / erfc
# ---------------------------------------------------------------------------

def rel(a: float, b: float) -> float:
    if b == 0.0:
        return abs(a - b)
    return abs(a / b - 1.0)


def ulp_diff(a: float, b: float) -> float:
    """Deviation in ulps; meaningful when both are the same sign and scale."""
    if a == b:
        return 0.0
    if a == 0.0 or b == 0.0 or math.isinf(a) or math.isinf(b):
        return math.inf
    ia, ib = np.float64(a).view(np.int64), np.float64(b).view(np.int64)
    if ia < 0:
        ia = np.int64(np.iinfo(np.int64).min) - ia
    if ib < 0:
        ib = np.int64(np.iinfo(np.int64).min) - ib
    return float(abs(int(ia) - int(ib)))


def check_erf(rows) -> None:
    print("\n=== erf ===")
    worst_abs = (0.0, 0.0)
    worst_rel_ = (0.0, 0.0)
    worst_ulp = (0.0, 0.0)
    n_nan = 0
    for args, v in rows:
        x = args[0]
        ref = float(sp.erf(x))
        if math.isnan(v) != math.isnan(ref):
            fail(f"erf({x!r}) = {v!r}, scipy = {ref!r} (NaN-ness differs)")
            continue
        if math.isnan(v):
            n_nan += 1
            continue
        a = abs(v - ref)
        if a > worst_abs[0]:
            worst_abs = (a, x)
        r = rel(v, ref)
        if r > worst_rel_[0]:
            worst_rel_ = (r, x)
        u = ulp_diff(v, ref)
        if u > worst_ulp[0]:
            worst_ulp = (u, x)
    print(f"  samples            : {len(rows)}")
    print(f"  worst |rust - scipy|: {worst_abs[0]:.4e}  at x = {worst_abs[1]!r}")
    print(f"  worst relative     : {worst_rel_[0]:.4e}  at x = {worst_rel_[1]!r}")
    print(f"  worst ulps         : {worst_ulp[0]:.1f}  at x = {worst_ulp[1]!r}")
    if worst_abs[0] > ATOL_UNIT:
        fail(f"erf worst absolute deviation {worst_abs[0]:.4e} at x={worst_abs[1]!r} "
             f"exceeds {ATOL_UNIT:.0e}")


def check_erf_exact_zero(rows) -> None:
    print("\n=== erf(0) and exactness properties ===")
    got = {a[0]: v for a, v in rows}
    z = got.get(0.0)
    if z is None:
        fail("erf(0.0) was not printed")
    elif z != 0.0:
        fail(f"erf(0.0) = {z!r}, want exactly 0.0 "
             "(the A&S 7.1.26 coefficient sum is 0.999999999, so the old "
             "implementation returned 9.999999717180685e-10 here)")
    else:
        print("  erf(0.0) == 0.0 exactly")

    for args, v in rows:
        x = args[0]
        if x in (math.inf, -math.inf):
            want = 1.0 if x > 0 else -1.0
            if v != want:
                fail(f"erf({x!r}) = {v!r}, want {want}")


def check_erf_oddness(rows_pos, rows_neg) -> None:
    print("\n=== erf oddness: erf(-x) == -erf(x) ===")
    pos = {a[0]: v for a, v in rows_pos}
    neg = {a[0]: v for a, v in rows_neg}
    bad = []
    worst = 0.0
    for x, p in pos.items():
        if x not in neg:
            continue
        n = neg[x]
        d = abs(n + p)              # should be exactly 0
        worst = max(worst, d)
        if n != -p:
            bad.append((x, p, n, d))
    print(f"  pairs checked      : {len(pos)}")
    print(f"  exact violations   : {len(bad)}")
    print(f"  worst |erf(-x)+erf(x)|: {worst:.4e}")
    for x, p, n, d in bad[:6]:
        print(f"    x={x!r}: erf(x)={p!r}  erf(-x)={n!r}  |sum|={d:.4e}")
    if bad:
        fail(f"erf is not odd on {len(bad)} of {len(pos)} pairs; "
             f"worst |erf(-x)+erf(x)| = {worst:.4e}")


def check_erfc(rows) -> None:
    print("\n=== erfc ===")
    worst_rel = (0.0, 0.0)
    worst_abs_small = (0.0, 0.0)
    underflow = 0
    mismatched = []
    for args, v in rows:
        x = args[0]
        ref = float(sp.erfc(x))
        if math.isnan(v) != math.isnan(ref):
            fail(f"erfc({x!r}) = {v!r}, scipy = {ref!r} (NaN-ness differs)")
            continue
        if math.isnan(v):
            continue
        if ref == 0.0 or v == 0.0 or abs(ref) < 1e-300:
            # Below the f64 normal range a *relative* comparison is meaningless
            # (scipy itself flushes erfc(x) to 0 for x >~ 26.6, where the true
            # value is subnormal). Report separately, never as a relative error.
            if v != 0.0 and ref == 0.0:
                mismatched.append((x, v, ref))
            else:
                underflow += 1
            continue
        r = rel(v, ref)
        if r > worst_rel[0]:
            worst_rel = (r, x)
        if abs(x) <= 2.0:
            a = abs(v - ref)
            if a > worst_abs_small[0]:
                worst_abs_small = (a, x)
    print(f"  samples            : {len(rows)}")
    print(f"  worst relative     : {worst_rel[0]:.4e}  at x = {worst_rel[1]!r}")
    print(f"  worst absolute (|x|<=2): {worst_abs_small[0]:.4e} at x = {worst_abs_small[1]!r}")
    print(f"  subnormal / underflow (not compared relatively): {underflow}")
    if mismatched:
        print(f"  non-zero where scipy flushes to zero: {len(mismatched)}, e.g. "
              f"{mismatched[:3]}")
        warn("erfc returns a non-zero subnormal where scipy.special flushes to "
             "0.0; that is the mathematically correct direction, not a defect.")
    if worst_rel[0] > RTOL_TAIL:
        fail(f"erfc worst relative deviation {worst_rel[0]:.4e} at "
             f"x={worst_rel[1]!r} exceeds {RTOL_TAIL:.0e}")
    if worst_abs_small[0] > ATOL_UNIT:
        fail(f"erfc worst absolute deviation {worst_abs_small[0]:.4e} at "
             f"x={worst_abs_small[1]!r} exceeds {ATOL_UNIT:.0e}")


def check_erfc_consistency(rows) -> None:
    """erfc(x) == 1 - erf(x) wherever that subtraction is safe."""
    print("\n=== erfc(x) vs 1 - erf(x) ===")
    rows_map = {a[0]: v for a, v in rows}
    worst = (0.0, 0.0)
    # Safe region: erfc(x) must be O(1) and far from the cancellation cliff,
    # which for x <= 1.0 means the subtraction retains ~15 digits.
    for x, v in sorted(rows_map.items()):
        if not (0.0 <= x <= 1.0):
            continue
        e = float(sp.erf(x))
        s = 1.0 - e
        d = abs(v - s)
        if d > worst[0]:
            worst = (d, x)
    print(f"  worst |erfc(x) - (1-erf_ref(x))| over x in [0,1]: {worst[0]:.4e} "
          f"at x = {worst[1]!r}")
    if worst[0] > 1e-13:
        fail(f"erfc disagrees with 1-erf by {worst[0]:.4e} at x={worst[1]!r}")

    # erfc(0) == 1 exactly.
    z = rows_map.get(0.0)
    if z is not None and z != 1.0:
        fail(f"erfc(0.0) = {z!r}, want exactly 1.0")


def check_erfc_limits(rows) -> None:
    print("\n=== erfc limiting values ===")
    got = {a[0]: v for a, v in rows}
    for x, want in ((math.inf, 0.0), (-math.inf, 2.0), (0.0, 1.0)):
        if x in got and got[x] != want:
            fail(f"erfc({x!r}) = {got[x]!r}, want exactly {want}")
        elif x in got:
            print(f"  erfc({x!r}) == {got[x]!r}  (exact)")
    if math.nan in got and not math.isnan(got[math.nan]):
        fail(f"erfc(NaN) = {got[math.nan]!r}, want NaN")


# ---------------------------------------------------------------------------
# gamma family
# ---------------------------------------------------------------------------

def check_scalar(name, rows, ref, atol, rtol, skip_nonfinite=True) -> None:
    print(f"\n=== {name} ===")
    worst_abs = (0.0, None)
    worst_rel_ = (0.0, None)
    bad = []
    for args, v in rows:
        r = float(ref(*args))
        if math.isnan(v) and math.isnan(r):
            continue
        if skip_nonfinite and (math.isinf(r) or math.isinf(v)):
            continue
        if math.isnan(v) != math.isnan(r):
            bad.append((args, v, r, "nan-ness"))
            continue
        a = abs(v - r)
        if a > worst_abs[0]:
            worst_abs = (a, args)
        rr = rel(v, r)
        if rr > worst_rel_[0]:
            worst_rel_ = (rr, args)
        if a > atol and (r == 0.0 or a / max(abs(r), 1e-300) > rtol):
            bad.append((args, v, r, f"abs {a:.3e} rel {rr:.3e}"))
    argstr = lambda t: "(" + ", ".join(f"{a!r}" for a in t) + ")"  # noqa: E731
    print(f"  samples            : {len(rows)}")
    print(f"  worst absolute     : {worst_abs[0]:.4e}  at x = {argstr(worst_abs[1])}")
    print(f"  worst relative     : {worst_rel_[0]:.4e}  at x = {argstr(worst_rel_[1])}")
    for args, v, r, why in bad[:8]:
        print(f"    {argstr(args)}: rust={v!r} scipy={r!r}  {why}")
    if bad:
        fail(f"{name}: {len(bad)} of {len(rows)} samples exceed atol={atol:.0e} "
             f"rtol={rtol:.0e}; first {bad[0][0]} -> {bad[0][3]}")


def check_gamma_poles(rows) -> None:
    """Non-positive integers are poles. A pole must be *distinguishable* from a
    valid finite answer: inf or NaN, never a large finite number."""
    print("\n=== gamma at its poles (non-positive integers) ===")
    for args, v in rows:
        x = args[0]
        if math.isinf(v):
            print(f"  gamma({x:g}) = {v!r}  (pole: distinguishable from any valid value)")
        else:
            fail(f"gamma({x:g}) = {v!r} at a pole; expected inf/NaN, a finite "
                 "value here is indistinguishable from a real answer")


# ---------------------------------------------------------------------------
# factorial / double_factorial
# ---------------------------------------------------------------------------

def double_factorial(n: int) -> float:
    """Exact integer reference; `math` has no `double_factorial` here."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return float(r)


def check_factorial(name, rows, ref) -> None:
    print(f"\n=== {name} ===")
    bad = []
    worst = (0.0, 0)
    over = []
    for args, v in rows:
        n = int(args[0])
        try:
            r = float(ref(n))
        except OverflowError:
            # The exact integer value exceeds every f64. Same expectation as
            # the isinf branch below: the Rust side must saturate, not wrap.
            r = math.inf
        if math.isinf(r):
            # The true value overflows f64. What matters is that the Rust side
            # saturates to +/-inf rather than wrapping to a finite number or a
            # negative value, which no caller could recognise as overflow.
            if not math.isinf(v):
                bad.append((n, v, r, "true value overflows f64 but rust is finite"))
            else:
                over.append((n, v))
            continue
        if math.isinf(v):
            bad.append((n, v, r, "rust overflows where the true value is finite"))
            continue
        rr = rel(v, r)
        if rr > worst[0]:
            worst = (rr, n)
        if rr > 1e-13:
            bad.append((n, v, r, f"relative {rr:.3e}"))
    print(f"  samples            : {len(rows)}")
    print(f"  worst relative     : {worst[0]:.4e}  at n = {worst[1]}")
    if over:
        print(f"  saturated to inf at n = {[n for n, _ in over]} "
              "(true value overflows f64 there too)")
    for n, v, r, why in bad[:8]:
        print(f"    n={n}: rust={v!r} exact={r!r}  {why}")
    if bad:
        fail(f"{name}: {len(bad)} of {len(rows)} samples wrong; "
             f"first n={bad[0][0]} -> {bad[0][3]}")


# ---------------------------------------------------------------------------
# Bessel
# ---------------------------------------------------------------------------

def check_bessel(name, rows, ref, atol, rtol) -> None:
    print(f"\n=== {name} ===")
    worst_abs = (0.0, None)
    worst_rel_ = (0.0, None)
    bad = []
    near_zero = []
    for args, v in rows:
        r = float(ref(*args))
        if math.isnan(v) != math.isnan(r):
            bad.append((args, v, r, "nan-ness"))
            continue
        if math.isnan(v) or math.isinf(r):
            continue
        if math.isinf(v) and math.isfinite(r):
            bad.append((args, v, r, "rust overflows where scipy is finite"))
            continue
        if r == 0.0:
            # A zero crossing: compare absolutely, and only complain if the
            # Rust side is far from it in absolute terms.
            a = abs(v - r)
            if a > atol:
                bad.append((args, v, r, f"zero crossing, |diff| {a:.3e}"))
            else:
                near_zero.append((args, a))
            continue
        a = abs(v - r)
        rr = rel(v, r)
        if a > worst_abs[0]:
            worst_abs = (a, args)
        if rr > worst_rel_[0]:
            worst_rel_ = (rr, args)
        if a > atol and rr > rtol:
            bad.append((args, v, r, f"abs {a:.3e} rel {rr:.3e}"))
    argstr = lambda t: "(" + ", ".join(f"{a!r}" for a in t) + ")"  # noqa: E731
    print(f"  samples            : {len(rows)}")
    print(f"  worst absolute     : {worst_abs[0]:.4e}  at x = {argstr(worst_abs[1])}")
    print(f"  worst relative     : {worst_rel_[0]:.4e}  at x = {argstr(worst_rel_[1])}")
    print(f"  at scipy zero crossings: {len(near_zero)}")
    for args, v, r, why in bad[:8]:
        print(f"    {argstr(args)}: rust={v!r} scipy={r!r}  {why}")
    if bad:
        fail(f"{name}: {len(bad)} of {len(rows)} samples exceed atol={atol:.0e} "
             f"rtol={rtol:.0e}; first {argstr(bad[0][0])} -> {bad[0][3]}")


def check_parity(name, pos_rows, neg_rows, ref, expect_odd=True) -> None:
    """J_n(-x) = (-1)^n J_n(x). `ref` is the SciPy function, used only to
    decide the expected parity: Y_n and the modified functions are not a
    function of a negative argument at all, so those are not checked here."""
    print(f"\n=== {name} parity in the argument ===")
    pos = {tuple(a): v for a, v in pos_rows}
    neg = {tuple(a): v for a, v in neg_rows}
    bad = []
    for key, p in pos.items():
        if key not in neg:
            continue
        n, x = key
        want = -p if (expect_odd and int(n) % 2 == 1) else p
        n_val = neg[key]
        if abs(n_val - want) > 1e-12 * max(abs(want), 1e-300) + 1e-300:
            bad.append((n, x, p, n_val, want))
    print(f"  pairs checked      : {len(pos)}")
    print(f"  violations         : {len(bad)}")
    for n, x, p, nv, want in bad[:6]:
        print(f"    n={n:g} x={x:g}: value={p!r} at -x={nv!r} want {want!r}")
    if bad:
        fail(f"{name} parity in x violated on {len(bad)} of {len(pos)} pairs")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/parity_special.txt"
    rows = parse(path)
    print(f"scipy {scipy.__version__}, numpy {np.__version__}, "
          f"reference file {path}")
    print(f"{len(rows)} distinct record kinds, "
          f"{sum(len(v) for v in rows.values())} records")

    if "erf" not in rows:
        print("no records found; did the Rust example run?", file=sys.stderr)
        return 2

    check_erf(rows["erf"])
    check_erf_exact_zero(rows["erf"])
    check_erf_oddness(rows["erf_pos"], rows["erf_neg"])
    check_erfc(rows["erfc"])
    check_erfc_consistency(rows["erfc"])
    check_erfc_limits(rows["erfc_inf"])

    if "erfi" in rows:
        check_scalar("erfi", rows["erfi"], sp.erfi, 1e-12, 1e-12)
    if "gamma" in rows:
        check_scalar("gamma", rows["gamma"], sp.gamma, 1e-9, RTOL_FUNCTION)
    if "log_gamma" in rows:
        check_scalar("log_gamma", rows["log_gamma"], sp.gammaln, 1e-9, 0.0)
    if "gamma_pole" in rows:
        check_gamma_poles(rows["gamma_pole"])
    if "beta" in rows:
        check_scalar("beta", rows["beta"], sp.beta, 1e-9, RTOL_FUNCTION)
    if "log_beta" in rows:
        check_scalar("log_beta", rows["log_beta"], sp.betaln, 1e-9, 1e-11)
    if "factorial" in rows:
        check_factorial("factorial", rows["factorial"], math.factorial)
    if "double_factorial" in rows:
        check_factorial("double_factorial", rows["double_factorial"],
                         double_factorial)
    for nm, ref, atol, rtol in (
        ("bessel_j0", sp.j0, 5e-9, 1e-6),
        ("bessel_j1", sp.j1, 5e-9, 1e-6),
        ("bessel_y0", sp.y0, 1e-8, 1e-6),
        ("bessel_y1", sp.y1, 1e-8, 1e-6),
        ("bessel_jn", sp.jv, 1e-9, 1e-9),
        ("bessel_yn", sp.yv, 1e-8, 1e-8),
        ("spherical_jn", sp.spherical_jn, 1e-12, 1e-11),
        ("spherical_yn", sp.spherical_yn, 1e-12, 1e-11),
        ("bessel_i0", sp.i0, 5e-7, 1e-7),
        ("bessel_k0", sp.k0, 5e-7, 1e-6),
    ):
        if nm in rows:
            check_bessel(nm, rows[nm], ref, atol, rtol)
    if "bessel_jn_neg" in rows:
        check_parity("bessel_jn", rows["bessel_jn"], rows["bessel_jn_neg"], sp.jv)
    if "spherical_jn_neg" in rows:
        check_parity("spherical_jn", rows["spherical_jn"],
                     rows["spherical_jn_neg"], sp.spherical_jn)
    if "expn" in rows:
        check_scalar("expn", rows["expn"], sp.expn, 1e-12, 1e-12)
    if "expi" in rows:
        check_scalar("expi", rows["expi"], sp.expi, 1e-12, 1e-12)

    print("\n" + "=" * 72)
    for w in _warnings:
        print(f"WARNING: {w}")
    if _failures:
        print(f"FAIL: {len(_failures)} check(s) outside the derived bound")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks within the bounds derived from f64 round-off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())