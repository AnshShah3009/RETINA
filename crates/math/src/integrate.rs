//! Numerical integration: adaptive quadrature, fixed rules and an RK4 ODE
//! stepper.

/// 15-point Kronrod abscissae on `[-1, 1]` (non-negative half, descending),
/// QUADPACK `qk15`. Odd indices are the 7-point Gauss nodes.
const XGK: [f64; 8] = [
    0.991_455_371_120_812_639_206_854_697_526_329,
    0.949_107_912_342_758_524_526_189_684_047_851,
    0.864_864_423_359_769_072_789_712_788_640_926,
    0.741_531_185_599_394_439_863_864_773_280_788,
    0.586_087_235_467_691_130_294_144_845_693_013,
    0.405_845_151_377_397_166_906_606_412_076_961,
    0.207_784_955_007_898_467_600_689_403_773_245,
    0.0,
];
/// Kronrod weights for [`XGK`].
const WGK: [f64; 8] = [
    0.022_935_322_010_529_224_963_732_008_058_970,
    0.063_092_092_629_978_553_290_700_663_189_204,
    0.104_790_010_322_250_183_839_876_322_541_518,
    0.140_653_259_715_525_918_745_189_590_510_238,
    0.169_004_726_639_267_902_826_583_426_598_550,
    0.190_350_578_064_785_409_913_256_402_421_014,
    0.204_432_940_075_298_892_414_161_999_234_649,
    0.209_482_141_084_727_828_012_999_174_891_714,
];
/// 7-point Gauss weights for `XGK[1], XGK[3], XGK[5], XGK[7]`.
const WG: [f64; 4] = [
    0.129_484_966_168_869_693_270_611_432_679_082,
    0.279_705_391_489_276_667_901_467_771_423_780,
    0.381_830_050_505_118_944_950_369_775_488_975,
    0.417_959_183_673_469_387_755_102_040_816_327,
];

/// SciPy's default `epsabs` and `epsrel` for `scipy.integrate.quad`.
pub const QUAD_DEFAULT_TOL: f64 = 1.49e-8;
/// SciPy's default subinterval limit for `scipy.integrate.quad`.
pub const QUAD_DEFAULT_LIMIT: usize = 50;

/// One G7-K15 panel (QUADPACK `qk15`): `(integral, error estimate, ∫|f|, ∫|f - mean|)`. Only
/// interior points are evaluated. `Err` if `f` is non-finite at a node.
fn gk15(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> Result<(f64, f64, f64, f64), String> {
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let dhlgth = hlgth.abs();
    let eval = |x: f64| -> Result<f64, String> {
        let v = f(x);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(format!("quad: integrand is {v} at x = {x}"))
        }
    };

    let fc = eval(centr)?;
    let mut resg = fc * WG[3];
    let mut resk = fc * WGK[7];
    let mut resabs = resk.abs();
    let mut fv1 = [0.0; 7];
    let mut fv2 = [0.0; 7];
    for j in 0..7 {
        let absc = hlgth * XGK[j];
        let f1 = eval(centr - absc)?;
        let f2 = eval(centr + absc)?;
        fv1[j] = f1;
        fv2[j] = f2;
        let fsum = f1 + f2;
        resk += WGK[j] * fsum;
        resabs += WGK[j] * (f1.abs() + f2.abs());
        if j % 2 == 1 {
            resg += WG[j / 2] * fsum;
        }
    }
    let reskh = resk * 0.5;
    let mut resasc = WGK[7] * (fc - reskh).abs();
    for j in 0..7 {
        resasc += WGK[j] * ((fv1[j] - reskh).abs() + (fv2[j] - reskh).abs());
    }
    let result = resk * hlgth;
    resabs *= dhlgth;
    resasc *= dhlgth;
    let mut abserr = ((resk - resg) * hlgth).abs();
    if resasc != 0.0 && abserr != 0.0 {
        abserr = resasc * (200.0 * abserr / resasc).powf(1.5).min(1.0);
    }
    if resabs > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) {
        abserr = abserr.max(50.0 * f64::EPSILON * resabs);
    }
    Ok((result, abserr, resabs, resasc))
}

/// Adaptive Gauss-Kronrod quadrature of `f` over `[a, b]` with SciPy's
/// defaults (`epsabs = epsrel = 1.49e-8`, `limit = 50`).
///
/// Returns `(integral, estimated absolute error)`. See [`quad_with`].
///
/// This replaced a fixed 100-panel trapezoid rule that reported an error of
/// exactly 0 and evaluated `f` at the endpoints (`1/sqrt(x)` on `[0, 1]` gave
/// `inf`; SciPy 2).
pub fn quad(f: impl Fn(f64) -> f64, a: f64, b: f64) -> Result<(f64, f64), String> {
    quad_with(
        f,
        a,
        b,
        QUAD_DEFAULT_TOL,
        QUAD_DEFAULT_TOL,
        QUAD_DEFAULT_LIMIT,
    )
}

/// Adaptive G7-K15 quadrature with Wynn epsilon extrapolation - QUADPACK
/// `dqagse`, the routine behind `scipy.integrate.quad` for finite limits.
///
/// The subinterval with the largest error estimate is bisected until the
/// total estimate is `<= max(epsabs, epsrel * |integral|)`; the sequence of
/// partial sums is extrapolated with the epsilon algorithm, which is what
/// makes integrable endpoint singularities (`1/sqrt(x)`, `ln x`) converge in
/// a handful of subintervals. The rule is open - `f` is never evaluated at
/// `a` or `b`. Finite limits only; `a > b` negates, `a == b` is `(0, 0)`.
///
/// Returns `(integral, estimated absolute error)`. Where SciPy returns a
/// number with an `IntegrationWarning`, this returns `Err` with the current
/// estimate in the message: subdivision `limit` reached, roundoff detected,
/// bad integrand behaviour (a subinterval too small to bisect), extrapolation
/// roundoff, probable divergence, or a non-finite integrand value.
pub fn quad_with(
    f: impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Result<(f64, f64), String> {
    if !a.is_finite() || !b.is_finite() {
        return Err(format!("quad: limits must be finite, got [{a}, {b}]"));
    }
    if a == b {
        return Ok((0.0, 0.0));
    }
    if a > b {
        return quad_with(f, b, a, epsabs, epsrel, limit).map(|(v, e)| (-v, e));
    }
    if !(epsabs > 0.0 || epsrel >= 50.0 * f64::EPSILON) {
        return Err("quad: tolerance cannot be achieved with given epsabs and epsrel".into());
    }
    qags(&f, a, b, epsabs, epsrel, limit.max(1))
}

/// Subinterval store of `dqagse`, with the error ordering kept by re-sorting.
struct Workspace {
    a: Vec<f64>,
    b: Vec<f64>,
    r: Vec<f64>,
    e: Vec<f64>,
    level: Vec<usize>,
    /// Indices by decreasing error.
    order: Vec<usize>,
    nrmax: usize,
    /// Interval to bisect next.
    i: usize,
    maximum_level: usize,
}

impl Workspace {
    fn new(a: f64, b: f64, r: f64, e: f64) -> Self {
        Self {
            a: vec![a],
            b: vec![b],
            r: vec![r],
            e: vec![e],
            level: vec![0],
            order: vec![0],
            nrmax: 0,
            i: 0,
            maximum_level: 0,
        }
    }

    fn len(&self) -> usize {
        self.a.len()
    }

    /// Replace interval `i` by its two halves (larger error kept in place).
    #[allow(clippy::too_many_arguments)]
    fn update(&mut self, a1: f64, b1: f64, r1: f64, e1: f64, a2: f64, b2: f64, r2: f64, e2: f64) {
        let i = self.i;
        let new_level = self.level[i] + 1;
        let (keep, add) = if e2 > e1 {
            ((a2, b2, r2, e2), (a1, b1, r1, e1))
        } else {
            ((a1, b1, r1, e1), (a2, b2, r2, e2))
        };
        self.a[i] = keep.0;
        self.b[i] = keep.1;
        self.r[i] = keep.2;
        self.e[i] = keep.3;
        self.level[i] = new_level;
        self.a.push(add.0);
        self.b.push(add.1);
        self.r.push(add.2);
        self.e.push(add.3);
        self.level.push(new_level);
        self.maximum_level = self.maximum_level.max(new_level);

        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_by(|&x, &y| self.e[y].total_cmp(&self.e[x]));
        // As `qpsrt`: the bisected interval moves up past nrmax if its error
        // is now larger than those ahead of it.
        let pos = order.iter().position(|&k| k == i).unwrap_or(0);
        self.nrmax = self.nrmax.min(pos).min(order.len() - 1);
        self.order = order;
        self.i = self.order[self.nrmax];
    }

    fn large_interval(&self) -> bool {
        self.level[self.i] < self.maximum_level
    }

    fn increase_nrmax(&mut self, limit: usize) -> bool {
        let last = self.len() - 1;
        let jupbnd = if last > 1 + limit / 2 {
            (limit + 1).saturating_sub(last)
        } else {
            last
        };
        let mut k = self.nrmax;
        while k <= jupbnd && self.nrmax < self.order.len() {
            self.i = self.order[self.nrmax];
            if self.level[self.i] < self.maximum_level {
                return true;
            }
            self.nrmax += 1;
            k += 1;
        }
        false
    }

    fn reset_nrmax(&mut self) {
        self.nrmax = 0;
        self.i = self.order[0];
    }

    fn sum_results(&self) -> f64 {
        self.r.iter().sum()
    }
}

/// Wynn epsilon-algorithm table (QUADPACK `dqelg`).
struct EpsilonTable {
    n: usize,
    rlist2: [f64; 52],
    nres: usize,
    res3la: [f64; 3],
}

impl EpsilonTable {
    fn new() -> Self {
        Self {
            n: 0,
            rlist2: [0.0; 52],
            nres: 0,
            res3la: [0.0; 3],
        }
    }

    fn append(&mut self, y: f64) {
        if self.n < 52 {
            self.rlist2[self.n] = y;
            self.n += 1;
        }
    }

    /// Extrapolated limit and its error estimate.
    fn qelg(&mut self) -> (f64, f64) {
        let eps = f64::EPSILON;
        let epstab = &mut self.rlist2;
        let n = self.n - 1;
        let current = epstab[n];
        let mut relative = 5.0 * eps * current.abs();
        let newelm = n / 2;
        let n_orig = n;
        let mut n_final = n;
        let nres_orig = self.nres;
        let mut result = current;
        let mut abserr = f64::MAX;

        if n < 2 {
            return (current, f64::MAX.max(relative));
        }
        epstab[n + 2] = epstab[n];
        epstab[n] = f64::MAX;

        for i in 0..newelm {
            let mut res = epstab[n - 2 * i + 2];
            let e0 = epstab[n - 2 * i - 2];
            let e1 = epstab[n - 2 * i - 1];
            let e2 = res;
            let e1abs = e1.abs();
            let delta2 = e2 - e1;
            let err2 = delta2.abs();
            let tol2 = e2.abs().max(e1abs) * eps;
            let delta3 = e1 - e0;
            let err3 = delta3.abs();
            let tol3 = e1abs.max(e0.abs()) * eps;

            if err2 < tol2 && err3 < tol3 {
                // e0, e1, e2 equal to machine accuracy: converged.
                let absolute = err2 + err3;
                relative = 5.0 * eps * res.abs();
                return (res, absolute.max(relative));
            }

            let e3 = epstab[n - 2 * i];
            epstab[n - 2 * i] = e1;
            let delta1 = e1 - e3;
            let err1 = delta1.abs();
            let tol1 = e1abs.max(e3.abs()) * eps;

            if err1 < tol1 || err2 < tol2 || err3 < tol3 {
                n_final = 2 * i;
                break;
            }
            let ss = (1.0 / delta1 + 1.0 / delta2) - 1.0 / delta3;
            if (ss * e1).abs() <= 1e-4 {
                n_final = 2 * i;
                break;
            }
            res = e1 + 1.0 / ss;
            epstab[n - 2 * i] = res;
            let error = err2 + (res - e2).abs() + err3;
            if error <= abserr {
                abserr = error;
                result = res;
            }
        }

        let limexp = 50 - 1;
        if n_final == limexp {
            n_final = 2 * (limexp / 2) - 1;
        }
        if n_orig % 2 == 1 {
            for i in 0..=newelm {
                epstab[1 + i * 2] = epstab[i * 2 + 3];
            }
        } else {
            for i in 0..=newelm {
                epstab[i * 2] = epstab[i * 2 + 2];
            }
        }
        if n_orig != n_final {
            for i in 0..=n_final {
                epstab[i] = epstab[n_orig - n_final + i];
            }
        }
        self.n = n_final + 1;

        if nres_orig < 3 {
            self.res3la[nres_orig] = result;
            abserr = f64::MAX;
        } else {
            abserr = (result - self.res3la[2]).abs()
                + (result - self.res3la[1]).abs()
                + (result - self.res3la[0]).abs();
            self.res3la[0] = self.res3la[1];
            self.res3la[1] = self.res3la[2];
            self.res3la[2] = result;
        }
        self.nres = nres_orig + 1;
        (result, abserr.max(5.0 * eps * result.abs()))
    }
}

fn qags(
    f: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> Result<(f64, f64), String> {
    let eps = f64::EPSILON;
    let fail = |what: &str, value: f64, err: f64| -> Result<(f64, f64), String> {
        Err(format!(
            "quad: {what}: integral ~ {value:e}, estimated error {err:e}"
        ))
    };

    let (result0, abserr0, resabs0, resasc0) = gk15(f, a, b)?;
    let mut tolerance = epsabs.max(epsrel * result0.abs());
    if abserr0 <= 100.0 * eps * resabs0 && abserr0 > tolerance {
        return fail("roundoff error on the first attempt", result0, abserr0);
    } else if (abserr0 <= tolerance && abserr0 != resasc0) || abserr0 == 0.0 {
        return Ok((result0, abserr0));
    } else if limit == 1 {
        return fail("subdivision limit 1 reached", result0, abserr0);
    }

    let mut ws = Workspace::new(a, b, result0, abserr0);
    let mut table = EpsilonTable::new();
    table.append(result0);

    let mut area = result0;
    let mut errsum = abserr0;
    let mut res_ext = result0;
    let mut err_ext = f64::MAX;
    let mut correc = 0.0;
    let mut ertest = 0.0;
    let mut error_over_large_intervals = 0.0;
    let (mut roundoff_type1, mut roundoff_type2, mut roundoff_type3) = (0, 0, 0);
    let mut error_type = 0;
    let mut error_type2 = false;
    let mut ktmin = 0;
    let mut extrapolate = false;
    let mut disallow_extrapolation = false;
    let positive_integrand = result0.abs() >= (1.0 - 50.0 * eps) * resabs0;
    let mut iteration = 1;
    let mut to_compute_result = false;

    loop {
        let i = ws.i;
        let (a_i, b_i, r_i, e_i) = (ws.a[i], ws.b[i], ws.r[i], ws.e[i]);
        let current_level = ws.level[i] + 1;
        let a1 = a_i;
        let b1 = 0.5 * (a_i + b_i);
        let a2 = b1;
        let b2 = b_i;
        iteration += 1;

        let (area1, error1, _, resasc1) = gk15(f, a1, b1)?;
        let (area2, error2, _, resasc2) = gk15(f, a2, b2)?;
        let area12 = area1 + area2;
        let error12 = error1 + error2;
        let last_e_i = e_i;

        errsum += error12 - e_i;
        area += area12 - r_i;
        tolerance = epsabs.max(epsrel * area.abs());

        if resasc1 != error1 && resasc2 != error2 {
            let delta = r_i - area12;
            if delta.abs() <= 1e-5 * area12.abs() && error12 >= 0.99 * e_i {
                if !extrapolate {
                    roundoff_type1 += 1;
                } else {
                    roundoff_type2 += 1;
                }
            }
            if iteration > 10 && error12 > e_i {
                roundoff_type3 += 1;
            }
        }
        if roundoff_type1 + roundoff_type2 >= 10 || roundoff_type3 >= 20 {
            error_type = 2;
        }
        if roundoff_type2 >= 5 {
            error_type2 = true;
        }
        // Bad integrand behaviour: no room left to bisect.
        let tmp = (1.0 + 100.0 * eps) * (a2.abs() + 1000.0 * f64::MIN_POSITIVE);
        if a1.abs() <= tmp && b2.abs() <= tmp {
            error_type = 4;
        }

        ws.update(a1, b1, area1, error1, a2, b2, area2, error2);

        if errsum <= tolerance {
            to_compute_result = true;
            break;
        }
        if error_type != 0 {
            break;
        }
        if iteration >= limit - 1 {
            error_type = 1;
            break;
        }
        if iteration == 2 {
            error_over_large_intervals = errsum;
            ertest = tolerance;
            table.append(area);
            continue;
        }
        if disallow_extrapolation {
            continue;
        }
        error_over_large_intervals -= last_e_i;
        if current_level < ws.maximum_level {
            error_over_large_intervals += error12;
        }
        if !extrapolate {
            // Is the interval to bisect next among the largest?
            if ws.large_interval() {
                continue;
            }
            extrapolate = true;
            ws.nrmax = 1.min(ws.order.len() - 1);
            ws.i = ws.order[ws.nrmax];
        }
        if !error_type2 && error_over_large_intervals > ertest && ws.increase_nrmax(limit) {
            continue;
        }

        table.append(area);
        let (reseps, abseps) = table.qelg();
        ktmin += 1;
        if ktmin > 5 && err_ext < 1e-3 * errsum {
            error_type = 5;
        }
        if abseps < err_ext {
            ktmin = 0;
            err_ext = abseps;
            res_ext = reseps;
            correc = error_over_large_intervals;
            ertest = epsabs.max(epsrel * reseps.abs());
            if err_ext <= ertest {
                break;
            }
        }
        if table.n == 1 {
            disallow_extrapolation = true;
        }
        if error_type == 5 {
            break;
        }
        ws.reset_nrmax();
        extrapolate = false;
        error_over_large_intervals = errsum;

        if iteration >= limit {
            break;
        }
    }

    let (mut result, mut abserr) = (res_ext, err_ext);
    if !to_compute_result && err_ext != f64::MAX {
        let mut go_compute = false;
        let mut done = false;
        if error_type != 0 || error_type2 {
            if error_type2 {
                err_ext += correc;
                abserr = err_ext;
            }
            if error_type == 0 {
                error_type = 3;
            }
            if res_ext != 0.0 && area != 0.0 {
                if err_ext / res_ext.abs() > errsum / area.abs() {
                    go_compute = true;
                }
            } else if err_ext > errsum {
                go_compute = true;
            } else if area == 0.0 {
                done = true;
            }
        }
        if !go_compute && !done {
            let max_area = res_ext.abs().max(area.abs());
            if positive_integrand || max_area >= 0.01 * resabs0 {
                let ratio = res_ext / area;
                if !(0.01..=100.0).contains(&ratio) || errsum > area.abs() {
                    error_type = 6;
                }
            }
        }
        to_compute_result = go_compute;
    } else {
        to_compute_result = true;
    }
    if to_compute_result {
        result = ws.sum_results();
        abserr = errsum;
    }
    if error_type > 2 {
        error_type -= 1;
    }
    match error_type {
        0 => Ok((result, abserr)),
        1 => fail(
            &format!("subdivision limit {limit} reached"),
            result,
            abserr,
        ),
        2 => fail(
            "roundoff error prevents reaching the tolerance",
            result,
            abserr,
        ),
        3 => fail(
            "bad integrand behaviour (non-integrable singularity?)",
            result,
            abserr,
        ),
        4 => fail("roundoff error in the extrapolation table", result, abserr),
        _ => fail("the integral is probably divergent", result, abserr),
    }
}

pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * f(x);
    }

    sum * h / 3.0
}

pub fn trapezoid(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;

    for i in 1..n {
        sum += f(a + i as f64 * h);
    }

    sum * h
}

/// Maximum number of RK4 steps [`ode_solve`] will take before refusing.
pub const ODE_MAX_STEPS: usize = 10_000_000;

/// Fixed-step classical RK4 for `y' = f(t, y)` from `t_span.0` to `t_span.1`.
///
/// Returns every `(t, y)` including the initial state. Steps are `dt` long
/// except the last, which is shortened to land exactly on `t_span.1`; times
/// are `t0 + i·dt` (no accumulated drift).
///
/// Errors: `dt` not finite and positive, `t_span.1 < t_span.0`, non-finite
/// limits, or more than [`ODE_MAX_STEPS`] steps required. The old version
/// overshot (`0..1`, `dt = 0.3` ended at `t = 1.2`) and silently stopped after
/// 10 000 steps.
pub fn ode_solve(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_span: (f64, f64),
    dt: f64,
) -> Result<Vec<(f64, Vec<f64>)>, String> {
    let (t0, t1) = t_span;
    if !(dt.is_finite() && dt > 0.0) {
        return Err(format!(
            "ode_solve: dt must be finite and positive, got {dt}"
        ));
    }
    if !(t0.is_finite() && t1.is_finite()) || t1 < t0 {
        return Err(format!(
            "ode_solve: t_span must be finite with t_end >= t_start, got ({t0}, {t1})"
        ));
    }
    // Number of steps: the last one may be short. A remainder below a few
    // ulps of the span is rounding, not a step.
    let ratio = (t1 - t0) / dt;
    let mut n_steps = ratio.ceil();
    if n_steps - ratio > 1.0 - 4.0 * f64::EPSILON * ratio.max(1.0) && n_steps > 0.0 {
        n_steps -= 1.0;
    }
    if !(n_steps <= ODE_MAX_STEPS as f64) {
        return Err(format!(
            "ode_solve: {n_steps} steps of dt = {dt} needed for ({t0}, {t1}), more than \
             the limit {ODE_MAX_STEPS}"
        ));
    }
    let n_steps = n_steps as usize;

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut result = Vec::with_capacity(n_steps + 1);
    result.push((t, y.clone()));

    for i in 1..=n_steps {
        let t_next = if i == n_steps { t1 } else { t0 + i as f64 * dt };
        let h = t_next - t;
        let k1 = f(t, &y);
        let k2 = f(t + h / 2.0, &add_vec(&y, &scale_vec(&k1, h / 2.0)));
        let k3 = f(t + h / 2.0, &add_vec(&y, &scale_vec(&k2, h / 2.0)));
        let k4 = f(t + h, &add_vec(&y, &scale_vec(&k3, h)));

        for j in 0..y.len() {
            y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
        t = t_next;
        result.push((t, y.clone()));
    }

    Ok(result)
}

fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn scale_vec(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

pub fn cumtrapz(f: impl Fn(f64) -> f64, t: &[f64], initial: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(t.len());
    result.push(initial);
    let mut current = initial;

    for i in 1..t.len() {
        let dt = t[i] - t[i - 1];
        let avg = (f(t[i]) + f(t[i - 1])) / 2.0;
        current += avg * dt;
        result.push(current);
    }

    result
}
