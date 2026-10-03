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

**Which is "right" depends on the contract, and that is the open question.** They
are not interchangeable:

- **align-corners** (`x·(w−1)/(nw−1)`) pins the *corner* samples, so a 2x downscale
  and a 2x upscale are exact inverses at the endpoints. It is what many scientific
  libraries use.
- **half-pixel** (`(x+0.5)·w/nw − 0.5`) treats samples as pixel *centres*, which is
  what OpenCV, and graphics APIs generally, use. It avoids the half-pixel shift that
  align-corners introduces at non-integer scales.

Neither is more correct in the abstract. What *is* a defect is that a library
positioned as an OpenCV replacement makes the opposite choice **silently** — a user
compositing `resize` with `warp_affine` or a camera matrix gets a half-pixel
displacement they cannot see.

**Not changed here.** Switching to half-pixel would alter every existing caller and
would make `resize` a round-trip of `warp_affine` rather than of its own inverse.
The decision is a documented contract question, and the same choice has to be made
consistently across `imgproc`, `hal`'s CPU and GPU resize, and the WGSL shader — three
implementations that currently agree with each other and not with OpenCV.

**Action for whoever decides:** state the convention in the `Interpolation` doc
comment — it is already documented there for the *downscaling aliasing* consequence,
but not for the half-pixel placement itself — and add a test that pins the impulse
response, since that is the measurement which distinguishes the two conventions in a
single number.

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

