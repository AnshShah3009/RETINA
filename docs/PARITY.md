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

**The `linear`/`cubic`/`lanczos` resize cases deviate by 40-80 grey levels on
98-99% of pixels.** That magnitude is far too large to be rounding. It is either a
different kernel/phase convention or a genuine defect, and **I have not determined
which**. Do not read this as either.

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

