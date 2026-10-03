# PARITY — Rust CV vs OpenCV / SciPy

Status: partial. See the sections below.

---

## Filters (cv-imgproc vs OpenCV 4.13) — VERIFIED, one defect found and fixed

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

### Attributes of the remaining deviations — NOT established, do not read as defects

21 cases still deviate after the fix. **I have not determined which side is right for
any of them**, and the plausible explanations are all still open:

- **Border handling.** The deviations are concentrated at the image edge
  (`ramp` cases are 65-69% over tolerance; `impulse` cases are 0.4-0.9%, i.e. only
  the pixels within the kernel radius). A different border mode produces exactly
  this signature.
- **Kernel *size*.** The harness compares Rust's kernel size against
  `cv2.getGaussianKernel(5, sigma)`; where the Rust side chose a different size the
  filters are not comparable at all.
- **The ellipse structuring element is a genuine definitional difference**, already
  measured: Rust's `MorphShape::Ellipse` 5x5 has **21** support pixels, OpenCV's has
  **17**. OpenCV uses a radius-0.5 convention, the Rust code an inclusive radius. Both
  cases are reported rather than one being silently excluded.

**Anyone continuing this must attribute each remaining deviation before calling it a
defect.** A parity harness that cries wolf gets deleted, and these have not yet earned
the right to be called bugs.

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

## Geometry (cv-imgproc vs OpenCV 4.13) — RUNS, deviations NOT attributed

`resize` and `warpAffine` now run end to end. **`cv2.remap` is unusable in this
environment** — it returns `(-5:Bad argument)` for *every* argument combination,
verified on a 16x16 zero image with float32 and float64 maps and both
`INTER_NEAREST` and `INTER_LINEAR`. The binding exists; the overload does not resolve.
The remap block therefore **skips and says so** rather than raising, because a parity
report that silently omits a comparison is worse than one that fails.

### Deviations present, none of them attributed yet

```
resize_checker_nearest_13x40      max=160  2.5% over tol   rust=200 ref=40
resize_checker_linear_24x18       max=80  99.1% over tol   rust=40  ref=120
warpaffine_smooth_linear_c0       max=4   29.9% over tol   rust=100 ref=104
warpaffine_checker_linear_c0      max=4   43.5% over tol   rust=182 ref=186
```

**The `nearest` case is a coordinate-phase convention, not a defect.** Verified
directly: for a checkerboard downscaled with `INTER_NEAREST`, OpenCV's
`out[0,0]` picks the source pixel `img[0,0]` (value `0`) where the Rust side picks
the other phase (`200`). Both are self-consistent; they round a half-pixel boundary
in opposite directions. Which is "right" is a documented-choice question, not a bug.

**The `warpaffine` cases deviate by 1-4 grey levels** on a border mode that both
sides were told to use identically, with deviations concentrated away from the
interior. Small and localised, but **not yet attributed** — could be a fixed-point
rounding difference, a border-path detail, or a real disagreement.

### ATTRIBUTED: resize point-samples where OpenCV area-averages

The 40-80 grey level resize deviations are **not** a phase convention, and not
rounding. Established by measurement, for the 48x36 -> 24x18 case (an exact 2x
downscale) on the `smooth` input, row 0:

```
src row0     : 127 153 177 198 214 224 227 224 214 198 177 153 127
rust         : 127 178 215 227 211 172 120  71  37  28  47  88 141
cv2 LINEAR   : 148 194 222 224 200 157 106  60  33  31  54  97 148
cv2 INTER_AREA: 148 194 222 224 200 157 106  60  33  31  54  97 148
```

Three facts pin it:

1. **`rust[0] == src[0,0]` exactly.** The Rust side takes the source value at the
   pixel corner.
2. **OpenCV's `INTER_AREA` and `INTER_LINEAR` agree to `0.0`** on this downscale. The
   reference is self-consistent under two different algorithms.
3. Amplitudes match (peak-to-peak 199 rust vs 193 cv2), so both **do** filter — this is
   not nearest-neighbour. It is *where* the filter is centred.

So on a 2x downscale the Rust side **discards 3 of every 4 source pixels**, while
OpenCV averages the 2x2 neighbourhood. That is a genuine algorithmic difference and
it explains both the magnitude and the frequency dependence: on the checkerboard it
reads 40-80 apart, on the low-frequency `smooth` input only 11-21, because a
point-sample of a fast signal and a box average of it differ far more than either
differs from the other on a slow one.

**Not fixed here, deliberately.** This is a **semantics decision, not a bug fix**:

- `Interpolation::Linear` currently means "bilinear at the output pixel's mapped
  source coordinate", which is a defensible and widely used definition - it is what
  many libraries do, and it is correct for **upscaling**.
- For **downscaling** it is the wrong choice: bilinear point-sampling aliases, which
  is exactly why OpenCV offers `INTER_AREA` and why `INTER_AREA` and `INTER_LINEAR`
  agree here (area averaging *is* the right downscale filter).
- Changing `Linear` to area-average would fix the downscale case and alter every
  existing caller that upscales with it.

So the honest report is: the two libraries **disagree by design on downscaling**, and
the correct outcome is a decision - either match OpenCV, or document the divergence
and point users at a dedicated downscale path. What is now ruled out is the
possibility that either implementation has a rounding bug: the numbers are
self-consistent and reproducible on both sides.

Everything downstream of a resize inherits this, so it is worth deciding before it is
relied upon.

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
**I have not determined the cause and am not calling it a Rust defect.**

**Not attributable.** `epipole_E_true_unit_norm`, `epipole_F_fit_frobenius_norm`,
`pose_recovered_rotation_vs_ground_truth` and
`pose_recovered_translation_direction_vs_ground_truth` all deviate. For the epipoles
and the pose these are the classic gauge cases — E is defined up to sign and scale,
and a recovered pose has a 4-fold rotation ambiguity — so a naive norm or
elementwise comparison is meaningless without a normalisation the harness does not
currently apply. They need the same treatment the Sampson fix just received.

### Not finished

An `f32` mask branch and the last sections were still in progress when the agent hit
its turn limit. (Note this section predates the later independent audits of
`calib3d`; see the bug log for the hand-eye and P3P work, where a `+`/`-` sign flip
in the translation accumulation was caught by mutation at a 0.92 m error.) **No calib3d result is claimed beyond the RANSAC residuals above**,
and the deviations listed as "not attributable" must not be read as defects until
their normalisation is fixed.

