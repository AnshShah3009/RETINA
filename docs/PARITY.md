# PARITY — Rust CV vs OpenCV / SciPy

Status: partial. See the sections below.

---

## Filters (cv-imgproc vs OpenCV 4.13) — VERIFIED, two defects found and fixed

**Environment:** OpenCV 4.13.0, numpy 2.4.6, Linux, Rust with SIMD.

The four inputs are regenerated independently on both sides and confirmed
**byte-identical** (constant 128, impulse at (32,24), unit-slope ramp,
3.0x2.0-cycle sinusoid), so every deviation below is the filter, not the input.

### DEFECT FOUND AND FIXED: a normalised blur did not preserve a constant

`gauss_const` matched OpenCV at sigma 0.75 and 1.5 and deviated at sigma 2.0 — on
**100% of pixels**, `127` where OpenCV gives `128`:

```
| gauss_const_s2.0_reflect101 | 0.5 | 1 | 1 | 3072/3072 (100.0%) | DEVIATES |
  rust = 127, ref = 128
```

A constant must survive a normalised blur. It did not, because the f32 kernel did
not sum to 1. Measured, for a 13-tap sigma-2.0 kernel:

```
f64 sum of the f32 taps : 0.999999994139216142   (5.9e-9 — the taps themselves)
f32 sum of the f32 taps : 1.000000357627868650   (3.6e-7 — the summation)
```

`128 * 0.99999964` truncates to `127`. So any detector thresholding a blurred image
saw a uniform brightness shift.

**Two fixes tried; the first did not work and that is worth recording.** Accumulating
the weights and their sum in f64 — the obvious repair — leaves the error at the *same*
`3.6e-7`, because it is the summation of the narrowed taps that carries it, not the
accumulation of the weights. What works is correcting the **last** tap to `1.0` minus
the f32 sum of the others: one subtraction, landing on the smallest tap in the
kernel (`4.5e-6` at the 13x13 corner, against a peak near 1), so the filter shape is
perturbed negligibly. After it every kernel measured sums to exactly `1.0`, and all
six `gauss_const` cases match.

Applied to both `gaussian_kernel` and `gaussian_kernel_1d`.

### Attributes of the remaining deviations — ALL attributed

The 21 cases split into three families, and none of them is a filter defect.

**Quantisation truncation — a real defect, found and fixed.** The Rust path
quantised the f32 convolution with `as u8`, which *truncates*, where OpenCV
rounds to nearest (`cvRound`, ties to even). A normalised blur of a non-integer
value is then systematically up to 1 LSB dark on about half the pixels — the
same truncation class that hid inside the constant-preservation defect above.
The attribution probe settles it: applying the library's own kernel coefficients
in f64 and rounding matches cv2 **exactly** on `const` and `impulse`, so on
those inputs the entire discrepancy was the quantisation step. Fixed at every
float-to-u8 output site in `imgproc` (convolve, edges, geometry/warp, resize,
color) with `round_ties_even` — `cvRound` semantics. The one test that had
encoded a truncated golden value (127 where the sample lands on 127.5) moved
with it. 12 of the 18 gaussian cases closed; over-tolerance counts elsewhere
halved (e.g. `warpaffine_smooth_linear_c0` 29.9% → 10.4%).

**The remaining 6 gaussian cases (±1 LSB) are OpenCV's fixed-point path.** The
same probe reports `max|rust_kernel_in_f64_rounded - cv2| = 1` on `ramp`/
`sinusoid` and at sigma 2.0 on `impulse` — even exact f64 arithmetic with the
*same kernel* disagrees with cv2 by 1 count there, because OpenCV's 8u
`GaussianBlur` builds a fixed-point kernel and rounds its intermediate passes.
An f32 implementation cannot match that bit for bit; the ±1 LSB on
high-contrast inputs is the reference's arithmetic, not a Rust error.

**The 3 morphological cases are a definitional difference, measured.** Rust's
`MorphShape::Ellipse` 5x5 has **21** support pixels, OpenCV's **17** (inclusive
radius vs radius-0.5). Morphology is exact min/max over integers on both sides —
they simply use different structuring elements. Both the same-kernel and
OpenCV-kernel cases are reported rather than one being silently excluded.

## Special functions (cv-math vs SciPy 1.17) — VERIFIED, prediction confirmed

The `erf` accuracy gap recorded in `docs/bug-log.md` from the A&S 7.1.26 coefficient
sum was **confirmed by measurement**, not left as a reading:

```
214 samples, x in [-4, 4]
worst |rust - scipy.special.erf| = 1.3851e-07   at x = -1.4
within the A&S 7.1.26 bound of 2e-7: yes
```

Nine orders of magnitude behind SciPy, as predicted, and no worse. `erf` was
deliberately **not** changed.

## Geometry (cv-imgproc vs OpenCV 4.13) — RUNS, deviations attributed

`resize` and `warpAffine` now run end to end. **`cv2.remap` is unusable in this
environment** — it returns `(-5:Bad argument)` for *every* argument combination,
verified on a 16x16 zero image with float32 and float64 maps and both
`INTER_NEAREST` and `INTER_LINEAR`. The binding exists; the overload does not resolve.
The remap block therefore **skips and says so** rather than raising, because a parity
report that silently omits a comparison is worse than one that fails.

### Where it stands after the half-pixel switch

The coordinate-contract question below was **decided: half-pixel**, and the
resize divergence closed with it. Before the switch the checker/linear cases were
off by up to 160 grey (a convention difference is a whole phase, not a few
levels); after it every resize case matches or differs only by the reference's
own arithmetic. The `nearest` rule turned out to be a third thing entirely -
measured against cv2 over 293 samples of 12 (w, nw) pairs, `INTER_NEAREST` is
`floor(x / (nw/w))`, the scale as a **f32 reciprocal**. Every multiply form
(`(x*w)/nw`, `x*(w/nw)` in f32 and f64) fails exactly one tie case (36 -> 40,
y = 30: all give 27.0 where cv2 picks 26), and half-pixel fails 101 of 293 -
nearest is NOT half-pixel, only the interpolating modes are. Rust now matches
cv2's nearest exactly.

What remains in the report, all attributed and none a Rust defect:

- **Linear, ±1 LSB** (e.g. `resize_smooth_linear_96x72`, 11.7% over): Rust's
  bilinear is the exact float mapping (verified bit-for-bit in the warpaffine
  probe); `cv2.resize` computes its interpolation weights in fixed point
  (1/32 tables), worth up to 1 LSB - measured directly: exact float bilinear vs
  `INTER_LINEAR` differs on 607/6912 pixels by max 1, exactly the case's shape.
- **Cubic/Lanczos interior, ≤3 LSB on high-contrast inputs**: the Rust side is
  bit-exact a numpy mirror of its own kernel (Catmull-Rom / Lanczos-3, per-tap
  edge replicate, half-pixel positions - 0 pixels differ on all 4 warp cases and
  the 96x72 resize); cv2's kernels are fixed-point tables with integer
  accumulation. On smooth input the same class is ≤1 LSB.
- **Cubic/Lanczos borders, up to 13 LSB at corners**: cv2's edge extension is
  not reproducible from behaviour alone - it matches neither tap-replicate
  (13 off), nor position-clamp (37 off), nor zero-extend (90 off) exactly.
  OpenCV's `resize.cpp` border handling would have to be read to close this.
  The Rust side is exactly its documented kernel, so the residue is attributed
  to the reference's border handling, flagged open rather than called a defect.

**The `warpaffine` cases deviate by 1-4 grey levels — attributed, and not a Rust
defect.** Probed to the standard of the resize attribution below: an exact float
bilinear warp of the same inputs through the same f32-echoed matrix and border
mode, mirroring `get_pixel_bilinear_with_border`'s weight order and rounding to
nearest, matches the Rust output **bit-for-bit on all 1728 pixels of all four
cases** (smooth/checker × c0/replicate). The Rust side *is* the mathematical
bilinear warp; the entire 1-4 LSB remainder is OpenCV's `warpAffine` computing
its source map and bilinear weights in fixed point (~1/1024) — worth a few LSB
at checker edges (160-grey steps) and under 1 LSB on gentle gradients, exactly
the observed scaling. Before the truncation fix the over-counts were double
(`warpaffine_smooth_linear_c0` 29.9% → 10.4%), because Rust was then wrong by up
to 1 LSB as well.

### ATTRIBUTED: the resize divergence is align-corners vs half-pixel, not a kernel difference

An impulse probe settles it — a single lit pixel tells you exactly which source
pixels each side reads, with no hypothesis about phase. A 16x16 image with one
`255` at the centre, downscaled to 8x8 (an exact 2x reduction):

```
OpenCV NEAREST : 255          (picks the source pixel)
OpenCV LINEAR  :  64  = 255/4 (exactly the 2x2 box average)
OpenCV AREA    :  64  = 255/4 (identical, which is the key datum)

Rust NEAREST   : 255          (agrees)
Rust LINEAR    :  47
```

`cv2 LINEAR` and `cv2 AREA` agreeing exactly says OpenCV is area-averaging on a
downscale. The Rust value is not a box average — and it is **exactly** what a
correct bilinear produces under a *different coordinate convention*:

```
output x=4, source width 16 -> 8

  align-corners  fx = x·(w−1)/(nw−1)   = 8.5714   dx = 0.5714
                  2-D weight on the impulse = (1−0.5714)² = 0.1837
                  255 × 0.1837 = 46.84  -> 47     <- MEASURED

  half-pixel     fx = (x+0.5)·w/nw − 0.5 = 8.5000  dx = 0.5000
                  2-D weight = (1−0.5)² = 0.2500
                  255 × 0.25 = 63.75 -> 64           <- OpenCV
```

So both implementations are doing a bilinear interpolation; they disagree because
**`cv_hal`'s CPU resize maps with `align-corners` and OpenCV maps with
`half-pixel`.** The Rust arithmetic is internally consistent — this is a convention
difference, not a bug.

**RESOLVED — the contract is half-pixel.** Neither convention is more correct in
the abstract; what decided it is that a library positioned as an OpenCV
replacement must not make the opposite choice **silently** — a user compositing
`resize` with `warp_affine` or a camera matrix would get a half-pixel
displacement they cannot see. Half-pixel is what `cv2.resize` and graphics APIs
use, and interchangeability with OpenCV is worth more here than align-corners'
endpoint-inverse property.

Landed consistently across **all** implementations: `imgproc`'s cubic/lanczos
resampler and RGB linear, `hal`'s CPU resize, and the five WGSL resize shaders
(u8, f16, bf16, f32, f32-multichannel). `crates/imgproc/tests/resize_sampling_convention.rs`
pins the impulse response at **64 = 255/4** — OpenCV's value — and would have
caught either convention being changed alone. The GPU path remains guarded only
by code reading (no adapter test), so the "all backends move together"
requirement lives in that test's doc comment.

Measured after the switch: `resize_checker_linear_24x18` and `_96x72` — the
largest divergences in this section, up to 160 grey — now match exactly.

## signal_proc (cv-signal vs SciPy 1.17.1) — MATCHES

`crates/signal_proc/examples/sp_parity.rs` + `parity/parity_signal.py`. The Rust
side prints, SciPy recomputes, the two are compared. Deterministic 400-sample
input at 1 kHz — two sinusoids plus a DC offset, so the filter has real frequency
content to reject rather than a degenerate constant.

| order | cutoff | max abs err (b) | max abs err (a) | max abs err (filtfilt) |
|---:|---:|---:|---:|---:|
| 2 | 50 Hz | 1.388e-17 | 0.000e+00 | 6.253e-13 |
| 4 | 50 Hz | 6.202e-17 | 4.441e-16 | 5.883e-11 |
| 4 | 120 Hz | 5.551e-17 | 6.661e-16 | 1.421e-12 |

`butter` agrees with SciPy to the **last bit** — the coefficient errors are f64
round-off, not approximation error. `filtfilt` agrees to ~1e-11 absolute on values
of order 127, i.e. ~1e-13 relative.

**This is the same function whose start-up transient was fixed earlier today.** The
old code left a constant image coming back with max deviation **3.249 out of 3.25**,
because the forward pass's transient was mirrored by the reverse pass and survived
inside the unpadded region. A hand-computed constant test caught that defect; this
checks the general case against the reference implementation, which is the
stronger statement — a constant test proves one input, and this proves the filter
matches on arbitrary content.

Worth recording as a *result*: this is the first subsystem in the workspace verified
against SciPy at machine precision. The SciPy-named target is no longer a claim
about `math` alone.

## calib3d projection (cv-calib3d vs `cv2.projectPoints` 4.13.0) — MATCHES

`crates/calib3d/examples/parity_project.rs` + `parity/parity_calib_project.py`.
144 projections: 48 points at radii 0.2/0.6/1.0/1.5 through three distortion models,
with a non-trivial rotated and translated pose.

| distortion | max abs err u (px) | max abs err v (px) |
|---|---:|---:|
| none | 2.274e-13 | 1.137e-13 |
| mild (`k1=-0.28`) | 2.274e-13 | 2.274e-13 |
| strong (`k1=-0.82, k3=0.012`) | 1.137e-13 | 1.137e-13 |

**2.274e-13 px** on image coordinates of order 640 — about `3.5e-16` relative, which
is f64 round-off. The model, the pose convention and the radtan distortion all
agree with OpenCV exactly, including the strong case where the cubic radial term
dominates and a sign or coefficient-order error would be plainly visible.

The cloud deliberately spans a range of radii: at small `r` the radial terms are
near zero, so a bug in `k1`/`k2`/`p1`/`p2` would not show at all. The outer ring at
`r = 1.5` is what makes this comparison able to fail.

**This also settles the pose convention in this path.** `Pose` applies `R*p + t` and
`cv2.projectPoints` applies `R*X + t`; the two agree to round-off, so there is no
world-to-camera / camera-to-world ambiguity here. That ambiguity is the class of
defect this workspace keeps finding — a pose and its inverse are *both* well-formed,
so nothing downstream complains — and projection is where it would be most
misleading, because the wrong convention still produces plausible pixels.

Projection is the most-used operation in the workspace: every detection, every pose
estimate and every reprojection error passes through it.

## What parity is *possible* here, and what is not

Measured, because it bounds every claim in this report:

| reference library | installed | so |
|---|:--:|---|
| `cv2` (OpenCV) | **yes**, 4.13.0 | OpenCV-side parity is available |
| `scipy` | **yes**, 1.17.1 | SciPy-side parity is available |
| `numpy` | **yes**, 2.4.6 | shared |
| `open3d` | **yes**, 0.20.0 | Open3D-side parity now possible |
| `matplotlib` | **yes**, 3.11.2 | Matplotlib-side parity now possible |
| `sklearn` | **no** | no scikit-learn reference |

**All five reference stacks are now present**, so the earlier limit no longer
applies. What is *not* installed is `sklearn`, and VSLoc-RS has no Python reference
at all — for that target the comparison is against **source**, at
`/home/Phoenix/refs/visloc-rs`, recorded in `docs/VS_REFERENCE_COMPARISON.md`.

`sklearn` being absent means the nearest references for point-cloud geometry remain
`numpy` (exact arithmetic on hand-computed answers) and `scipy` (linear algebra,
`scipy.spatial`). That is what the `3d` and `registration` audits have used, and it
is why those results rest on hand-computed invariants rather than a reference.

`sklearn` being absent also means the nearest available references for
point-cloud geometry are `numpy` (exact arithmetic on hand-computed answers) and
`scipy` (linear algebra, `spatial.transform`). That is what the `3d` and
`registration` audits have been using, and it is why those results rest on
hand-computed invariants rather than on a reference implementation.

## `cv-math` special functions vs SciPy 1.17.1 — `erf`/`erfc` now at machine precision

`crates/math/examples/parity_special.rs` + `parity/parity_special.py`. This closes
the one gap recorded as OPEN for the SciPy target.

### `erf`: `1.3851e-07` → `6.6613e-16`

```
=== erf ===                            samples : 606
  worst |rust - scipy| : 6.6613e-16  at x = -1.6
  worst ulps           : 6.0
=== erf(0) and exactness properties ===
  erf(0.0) == 0.0 exactly
=== erf oddness: erf(-x) == -erf(x) ===
  pairs checked : 204    exact violations : 0
  worst |erf(-x)+erf(x)| : 0.0000e+00
=== erfc ===                            samples : 735
  worst relative : 1.0458e-13 at x = 25.5   (the tail, where cancellation lives)
  worst absolute (|x| <= 2) : 6.6613e-16
```

**A 2x10^8 improvement**, and the two properties that were silently false before
now hold: `erf(0)` is exactly `0` (it was `9.999999717180685e-10`, so `erf` was not
odd about zero) and oddness is **exact** across 204 pairs, worst violation
`0.0000e+00`.

### What changed

The Abramowitz & Stegun 7.1.26 rational approximation is gone — its five
coefficients sum to `0.999999999`, and that residue was the entire error. It is
replaced by a Maclaurin series for small `|x|` and a continued fraction for
`erfc` in the tail, where `1 - erf(x)` cancels catastrophically. The A&S form is
retained as a private helper with a comment recording why it is not the entry point,
rather than deleted — so the reason it was rejected is visible to the next reader.

### What is still open, honestly

The same harness reports **7 functions outside their derived bounds**, and the agent
was interrupted before setting per-function tolerances. On inspection these look
like each function's **algorithmic floor** rather than defects, but that is a claim
not yet checked per function:

| function | first exceedance | looks like |
|---|---|---|
| `gamma` | rel `1.009e-11` | near f64 accumulation over a wide range |
| `factorial` | rel `1.160e-11` at n=20 | Stirling series truncation |
| `bessel_y0` | rel `1.124e-06` | the NR approximation's own limit |
| `bessel_jn` / `bessel_yn` | rel `3.7e-09` / `1.0e-08` | series truncation |
| `bessel_k0` | rel `2.901e-06` | ditto |
| `expi` | abs `1.868e-01`, rel `7.29e-09` at x=20 | first omitted term at x=20 is ~2.3e-8, so this is the optimal truncation point |

### Derived algorithmic floors, measured

Each implementation's floor was identified from the *method*, then confirmed by
measurement over the whole sampled domain. These are the tolerances the harness
should gate on:

| function | method | measured worst | tolerance |
|---|---|---:|---:|
| `log_gamma` | Lanczos g=5, n=6 | `1.338e-10` abs | `1e-9` abs |
| `gamma` | `sign · exp(log_gamma)` | `1.336e-10` rel | `1e-9` rel |
| `factorial` | same path | `1.336e-10` rel | `1e-9` rel |
| `bessel_j0` | polynomial rational fit | `5.969e-05` rel | `1e-3` rel |
| `bessel_y0` | polynomial rational fit | `1.067e-05` rel | `1e-3` rel |
| `bessel_k0` | polynomial fit + `-ln(x/2)·I0(x)` | `1.627e-03` rel | `1e-2` rel |
| `bessel_jn` / `bessel_yn` | series / Miller downward | `2.3e-07` rel | `1e-5` rel |
| `expi` | series, truncated at the optimum | `7.29e-09` rel | `1e-7` rel |
| `erf` | Maclaurin + continued fraction | `6.66e-16` | `1e-14` |
| `erfc` | continued fraction | `1.05e-13` rel | `1e-11` rel |

**None of these is a defect, and saying so is the result.** The Lanczos bound is
~1e-10 and the two `log_gamma`/`gamma`/`factorial` errors are all the *same* 1.336e-10
because they share one code path. The Bessel fits are polynomial rational
approximations, whose accuracy is set by the fit and not by the arithmetic — `1.6e-3`
on `k0` is the fit, not a coding error. `expi`'s `7.29e-09` at x=20 is the point where
the series term underflows relative to the running sum.

**A caution worth recording.** While measuring this I computed `gamma`'s error as
**200%** and nearly reported a sign defect. The cause was mine: I compared against
`math.exp(gammaln(x))`, which overflows for large `x`, and read the overflow as a
mismatch. Measured properly, `gamma(-0.5) = -0.0600196` etc. match SciPy in **both sign
and magnitude** — `log_gamma` returns `log|Γ|` by design and `gamma` reapplies the
sign correctly. Fourth time this session a number stood for two different quantities.

**Now a gate.** With those bounds applied, `parity_special.py` reports
`All checks within the bounds derived from f64 round-off` and exits zero.

**Verified to discriminate.** Perturbing one Lanczos coefficient
(`76.18009172947146` -> `76.19009172947146`, a realistic transcription slip of one
digit in the ninth place) fails **144 of 144** `log_gamma` samples, and takes
`beta`, `log_beta` and `factorial` with it:

```
log_gamma: 144 of 144 samples exceed; first [-4.5]   -> abs 3.963e-04
beta:        8 of 13 samples exceed; first [0.5, 0.5] -> rel 2.486e-04
factorial:   4 of 13 samples wrong;  first n=20       -> rel 2.750e-04
```

That is the property a gate needs and a diagnostic does not: it now distinguishes
"as designed" from "regressed", having been calibrated on the method rather than on
whatever happened to pass.

`factorial` also gained a note: its reference is `float(math.factorial(n))`, itself
only f64-accurate, so it cannot be held to 1e-13 however good our side is.

## `crates/plot` vs Matplotlib 3.11.2 — TWO structural divergences, measured

The last of the five targets to get a comparison. **Rendered-image diffing was
deliberately not used**: Agg antialiases, so a pixel comparison against a
hand-written SVG disagrees for reasons that say nothing about either implementation.
Both properties below have an analytic answer, and both are checkable.

### 1. Autoscaling margins: 10% against Matplotlib's 5%

`crates/plot/src/export.rs:96-97` pads each axis by

```rust
let x_pad = (max_x - min_x).max(1.0) * 0.1;
let y_pad = (max_y - min_y).max(1.0) * 0.1;
```

— **10% per side, 20% total**, of `max(span, 1)`.

Matplotlib's default is `margins.x = margins.y = 0.05`. Measured:

```
data [0, 1]        span 1      matplotlib xlim [-0.05, 1.05]   5% per side
data [0, 100]      span 100    matplotlib xlim [-5, 105]      5% per side
data [-5, 5]       span 10     matplotlib xlim [-5.5, 5.5]    5% per side
```

**A factor of two, on every non-degenerate span.** Not a rounding disagreement — the
plots are framed differently, and the two libraries will never put the same data at the
same place on the same figure size.

Note the `max(span, 1.0)` clamp is doing real work the other way: for data with
`span < 1` the padding becomes `0.1` *absolute*, so a tiny range gets a large
relative margin. Matplotlib's margin is always relative to the span, so it shrinks with
the data. For a range of `1e-5` the two differ by five orders of magnitude.

### 2. There are no tick values at all

Emitted SVG from `Figure::scatter(...)` with 21 points spanning x in `[0, 100]`:

```xml
<text x="400" y="33" class="title"  text-anchor="middle">ticks</text>
<text x="400" y="590" class="label"  text-anchor="middle">X</text>
<text x="15"  y="300" class="label"  text-anchor="middle" transform="rotate(-90, 15, 300)">Y</text>
<text x="688" y="70"  class="legend">s</text>
```

A title, axis labels and a legend — and **no numeric ticks on either axis, and no
tick marks**. `grep -rn "tick" crates/plot/src/` returns nothing: there is no tick
generation in the crate. Matplotlib emits them by default:

```
data [0, 1]      -> ticks [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
data [0, 100]    -> ticks [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
data [3.14159, 3.14160] -> ticks [3.14159, 3.141592, ..., 3.1416]
```

This is a **capability gap, not a convention difference**, and it is the one that
matters most for the Matplotlib target: an axis with no numbers on it is not a
plotting axis. Tick *placement* is well-defined — Matplotlib's `MaxNLocator` picks
from a scale, and that is a reference to compare against. It is also the natural next
task, and unlike the margin it is additive rather than a behaviour change.

### Not compared, and why

Rendering, marker shapes, line joins and font metrics — all of which depend on Agg's
antialiasing and font stack, and would produce differences that say nothing about the
implementation.

## Coverage of this report — and its holes

Done: filters (`imgproc`), geometry — resize and warpAffine only (`imgproc`),
special functions (`math`).
**Not reached:** `remap` (unusable here), `photo` tone mapping, and everything in
`calib3d`, `features`, `3d`, `registration`, `pointcloud`, `rendering`. So this is a
**partial** parity report covering 2 of ~30 crates, and nothing here should be read as
a statement about the rest.

The geometry section is the least mature part of this report: it runs, it produces
numbers, and **most of those numbers are not yet explained**. That is stated rather
than smoothed over, because the alternative — a table of unexplained deviations
labelled as findings — is exactly what makes a parity harness get ignored.

---

## calib3d — HARNESS RUNS. Results partially attributable; one section incomplete.

`parity/parity_calib.py` now runs end to end, and I fixed **two real bugs in the
harness itself** that would both have been reported as Rust defects:

### Harness bug 1: `sampson` double-transposed F

```python
ex1 = x1 @ m.T      # was
etx2 = x2 @ m.T     # was
```

Applied `m.T` to **both** vectors, which double-transposes `F` and produces a
quantity that is not an epipolar residual at all. Verified on a clean synthetic set
with `cv2.findFundamentalMat`: the old expression gave `0.827` where the correct
Sampson residual is ~1e-2 px. It also produced a *reference* of `157.65` against a
Rust model whose true residual was `5e-12` — which read as a large Rust defect and
was entirely the harness's error. F and E are gauge-ambiguous, so only the
normalised Sampson distance (scale-invariant by construction) is meaningful.

### Harness bug 2 (fixed by the agent): the 1-D comparison, plus shape mismatches

Carried forward from the previous attempt: a stale loop variable, a `ZEPIPOLES`
unpacking mismatch, numpy array-truthiness checks, a `tvec` arriving as a 9-vector,
and three further shape mismatches.

### Where the results stand

**Attributable, and the Rust side is correct.** The RANSAC fundamental and essential
comparisons report:

```
ransac_F_max_residual_on_clean: rust = 1.1e-11   reference = 156.561
ransac_E_max_residual_on_clean: rust = 1.2e-12   reference = 156.571
```

The Rust side recovers 16 inliers with epipolar residuals at the 1e-11 level — a
perfect fit on clean data, which is the correct answer. The reference figure of
~156 px on a 100-px-wide synthetic set is **not explicable as a pixel residual for
any model that fits the data**, and F and E agreeing to within 0.01 of each other
suggests the reference is computing something other than the intended quantity.
**Resolved - two harness defects at once.** The "clean" fit set is the whole
20-point contaminated set, so the comparison scored two *random samplers* against
each other; and cv2's model on this input is degenerate (singular values
[1.0009, 8.777e-4, 1.229e-18], epipole at infinity), whose Sampson residual on its
own fit points is exactly the ~156 px. Both rows were removed in favour of the
comparable quantities - inlier count and per-point inlier mask, which agree
exactly - with the measurements kept as a `ransac_quality_not_comparable` note.

**Resolved - all four were harness bugs, not defects.** The gauge reading was
wrong: the epipolar block read the `emit_pose` record `[t | R]` back to front,
feeding the translation to `rodrigues` and the rotation matrix's first row to the
translation slot. The wrong pose is self-consistent - its own E fits its own
points, so the residual cases stayed green - but `ep_pts2` was 243.337 px out and
every downstream "ground truth" was the nonsense pose. The gauge story was also
wrong in its own right: E from *known extrinsics* is a formula, not a fit, so
`epipole_E_true_unit_norm` (now `epipole_E_true_from_extrinsics`) compares it
entrywise, and `epipole_F_fit_frobenius_norm` asserted the *raw* estimator record
where the scaled one is the contract (raw ‖F‖ is 0.515; no contract fixes it).
With the record read correctly: `ep_pts2` to 1.1e-13 px, E bit-exact, recovered
pose vs ground truth 3.7e-15. `ep_resid_Ffit8`'s residual *equality* was the same
class - two independent solvers' fit quality - and became a Rust-own closure gate
with cv2's numbers kept in the note.

### Not finished

An `f32` mask branch and the last sections were still in progress when the agent hit
its turn limit. (Note this section predates the later independent audits of
`calib3d`; see the bug log for the hand-eye and P3P work, where a `+`/`-` sign flip
in the translation accumulation was caught by mutation at a 0.92 m error.) **No calib3d result is claimed beyond the RANSAC residuals above**,
and the deviations formerly listed as "not attributable" were harness bugs and are
closed above; the two divergences that remain in the report are the documented
legitimate differences (`undistort_radtan_inverse_answer`, `zhang_K`).

---

## features (ORB) — invariants only; descriptors are NOT comparable

`crates/features/tests/orb_detector_parity.rs`. Descriptors **cannot** be compared
against OpenCV and pretending otherwise is the classic way to make a parity report
useless: `Descriptor.data` is `Vec<u8>` — quantised bytes — and a correct ORB
rotates the patch by the dominant orientation and packs orientation bits into the
top of the keypoint. Two correct implementations produce different bit patterns for
the same corner.

What is comparable, and is measured: repeatability across calls, keypoint count
relative to the scene, in-image finite coordinates, one descriptor per keypoint, and
a uniform 32-byte dimension.

### The input size was the whole difficulty — a false failure I diagnosed as a defect

My first draft used a 64x64 synthetic image and the control test failed at 0.25
recall. The obvious reading was "ORB misses corners". Measured against OpenCV on the
same image:

```
  64x64   checkerboard -> cv2 ORB detections: 0
 128x128  checkerboard -> cv2 ORB detections: 76
 256x256  checkerboard -> cv2 ORB detections: 240
```

**OpenCV finds zero on the 64x64 image too.** ORB builds a scale pyramid with a
fixed level count and base scale, so below roughly 128 px nothing sits at a usable
scale. The failure was in the *premise* of my test, not in the detector — and the
Python was what distinguished them.

### Counts on the corrected 128x128 image

```
planted corners: 64
rust ORB:        205 detections
cv2 ORB:          30 detections
```

Both find structure where there is structure, which is the meaningful comparison; the
absolute counts differ, and that difference is **not attributed here**. Plausible
causes include the FAST threshold, the pyramid level count, and non-maximum
suppression radius — each of which changes the count without changing whether the
detector works. Anyone following this up should establish which before treating the
gap as a defect.

Five tests, all passing. The control (`orb_finds_the_planted_corners`) runs the
image-size reasoning rather than assuming it.

---

## `crates/3d` vs Open3D 0.20.0 — TWO REAL DIVERGENCES FOUND

`crates/3d/examples/parity_open3d.rs` + `parity/parity_open3d.py`. This is the
**first Open3D comparison possible in this workspace**, and it immediately found a
defect that no internal check could see.

### Outlier removal — MATCHES on all 10 configurations

| filter | parameters | rust n | open3d n | sets equal |
|---|---|---:|---:|:--:|
| statistical | nb=5, sr=2.0 | 27 | 27 | yes |
| statistical | nb=4, sr=2.0 | 27 | 27 | yes |
| statistical | nb=20, sr=2.0 | 27 | 27 | yes |
| radius | r=0.5, min=2 | 27 | 27 | yes |
| radius | r=0.12, min=4 | 19 | 19 | yes |

(all 10 rows match) Index sets are integers, so set equality is the whole comparison
and no tolerance is meaningful — none is applied.

### Voxel downsampling — legitimate difference, and it is NOT the centroid rule

Open3D returns the **arithmetic mean** of each voxel's points, and so does
`cv_3d::filters::voxel_downsample` and `spatial::VoxelGrid::downsample` — verified
by reading both implementations. A "first point in the voxel" implementation would
have shown up as a centroid distance of order the voxel size; the measured distances
are the half-voxel grid-anchor shift, at most `vs/2`. Counts differ where the two
disagree on which cells are occupied (`neg`: rust 5 vs Open3D 3 at vs=0.5), which is
a grid-anchor convention, not an arithmetic error.

### Normal estimation — CORRECTED: the 90 deg is Open3D's, not ours

The harness first reported a 90.0000 deg max angle and `min |n|-1 = 0.00e+00`, and I
read that as a zero-length normal in our code. **That reading was wrong.** Measured
both sides separately on the same 312-point sphere:

```
ours   (estimate_normals_knn): zero-length = []  nan = []
                            max angle to analytic = 6.2828 deg (k=20), 15.0000 (k=8)

Open3D estimate_normals    : zero-length = 0
                            max angle to analytic = 90.0000 deg (k=20), 90.0000 (k=8)
```

So:

- **Our normals are fine** — 6.3 deg and 15.0 deg worst case on a 12x24 sphere, no
  zero-length and no NaN output, and the eigenvector in
  `math::linalg::min_eigenvector_3x3` *is* normalised (line 334) with a fallback for
  the degenerate case. I had not read that far before claiming a defect.
- **Open3D produces the 90 deg**, from the sphere's poles where the k-NN
  neighbourhood is degenerate.

The harness conflated the two. `_norm_row` computed `max_d = max(ag.max(), ar.max(),
cross.max())` — taking the worst of *our* deviation, *the reference's* deviation, and
the angle between us — so a 90 deg belonging to Open3D was attributed to the
workspace. `min |n|-1` measured `got` after normalising, so the column was measuring
normalisation error, not length, and could not have shown a zero vector.

**Fixed in the harness:** the three deviations are now reported in separate columns so
a regression on either side is attributable. The plane case remains the control at
0.0000 deg on both.

This is the second time in this session that an unattributed deviation turned out to
be the *reference* being wrong rather than the implementation — the first was the
`resize` investigation, where an even-width input made two conventions identical. Both
were the same mistake: a single number standing for two different quantities.

### k-NN neighbour sets: ATTRIBUTED — a tie at the k-th boundary, not a defect

| knn | queries | sets equal | Jaccard | ordering equal |
|---:|---:|---:|---:|---:|
| 6 | 27 | NO | 0.8202 | 1/27 |
| 4 | 27 | NO | 0.7705 | 3/27 |

Initially unattributed. Investigated properly rather than accepted, and the answer
is that **the harness's own explanation was right and I doubted it wrongly.**

Query 0, sorted by distance:

```
idx   0  d = 0.000000
idx   1  d = 0.100000      idx 3 = 0.100000      idx 9 = 0.100000
idx   4  d = 0.141421      idx 10 = 0.141421     idx 12 = 0.141421
idx  13  d = 0.173205
```

With `k = 6` the k-th neighbour falls **inside a tie group**: three points sit at
*exactly* `0.141421`, and only two of them fit. Which two is arbitrary, so the set
is not determined by the distances at all. Every failing query shows the same shape —
one member substituted for another at an identical distance:

```
query 0: extra=[12] (d=0.141421)  missing=[10] (d=0.141421)
query 1: extra=[11] (d=0.141421)  missing=[3]  (d=0.141421)
query 2: extra=[14] (d=0.141421)  missing=[10] (d=0.141421)
```

**This is a defect in the harness's scene, not in either implementation.** A
cubic lattice at spacing 0.1 has enormous numbers of exact ties, and every
substitution is between equidistant points.

### The part that IS a real finding

Checked both implementations against **brute-force ground truth** rather than
against each other:

```
k=6:  cv_3d matches exact on  9/27      Open3D matches exact on 16/27
k=4:  cv_3d matches exact on 10/27      Open3D matches exact on 16/27
```

**Open3D's `KDTreeFlann` is approximate** — it returns a different k-th *distance*
from the exact one on 27 of 27 queries. So it is not a reference for exact k-NN at
all, and `cv_3d`'s deviation from it is not evidence about `cv_3d`.

`cv_3d`'s own misses are all tie-boundary choices too (9/27 is a low score *because*
most queries are ties, not because the search is wrong) — and on a scene with no
ties it does match brute force exactly. **Now verified**, in
`crates/3d/examples/knn_exact.rs`: 200 fixed-seed jittered points (all distances
distinct), brute-force ground truth computed in f64, neighbour *sets* compared
for k = 4, 6, 10:

```
k=4:  exact set on 200/200 tie-free queries
k=6:  exact set on 200/200 tie-free queries
k=10: exact set on 200/200 tie-free queries
```

The k-NN search itself is exact. Every disagreement with the lattice harness was
a tie at the k-th boundary, and every disagreement with Open3D is down to
Open3D's approximation, not ours.

### Also noted by the harness, in `crates/3d/src/gpu/registration.rs` — fixed, twice

The incremental rotation was built from a raw twist by **first-order
linearisation** (`inc[0][1] = -g; inc[0][2] = b; …`), which is the matrix
exponential only to first order, so the composed result was not a rotation:

```
det(R) - 1                          = 2.878e-03
|R[0][1] + R[1][0]|  (should be 0)  = 2.215e-05
```

Substituting a proper SE(3) exponential map (`exponential_map_se3`), everything
else identical, dropped that error to f64 round-off. **Fixed.**

**And the fix itself shipped a second defect, caught here on re-run.** The twist
was packed into the exponential map with its x and z rotation components swapped
(`Vector6::new(tx, ty, tz, g, b, a)` where the Jacobian solves
`x = [omega; t] = [a, b, g, tx, ty, tz]`). Any exp-map output is a valid
rotation, so det and orthogonality were perfect — but the rotation was in the
wrong plane, and `icp_point_to_plane` landed **5.234e-02** from the ground truth
where Open3D's `registration_icp` reached 9.378e-06 on the same clouds. The
packing is `(a, b, g)` now and the harness measures **8.396e-08** — tighter than
Open3D.

The lesson the repo keeps relearning: a shape assertion is not a correctness
assertion. `icp_returns_rigid_transform.rs` asserts rotation-ness and passed
under both defects; it now also asserts the **pose** against a known offset with
unequal x/z rotation components, and a mutation check confirms that test fails
under the swapped packing while the rotation-ness tests pass.

### Harness bugs found and fixed while getting this to run

Four, all of which would have produced a *false* result rather than an error:

1. `#CASE` was never handled, so `cur` stayed `None` and every per-case collection
   was filed under the wrong key.
2. `#NRM <k> <n>` assigned to the same variable as the case name, clobbering it with
   an integer.
3. `#VGP` has no count field, so slicing coordinates from the wrong index took two
   values instead of three and every point set failed a `reshape(-1, 3)`.
4. `compute_knn_graph` **does not exist in Open3D 0.20** (it raises
   `AttributeError`); the supported path is `KDTreeFlann.search_knn_vector_3d`.

Worth stating plainly: the agent that wrote this harness stopped at its turn limit
with it **not running at all**, and the failure surfaced as a `ValueError` rather
than a comparison. A parity script that raises is worse than no script, because it
looks like progress.
