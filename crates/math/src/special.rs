use std::f64::consts::PI;

pub fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

pub fn erfi(x: f64) -> f64 {
    if x < 0.0 {
        return -erfi(-x);
    }
    if x > 6.0 {
        // Asymptotic: e^{x^2}/(sqrt(pi)x) * sum (2k-1)!!/(2x^2)^k,
        // truncated at the smallest term (the series diverges).
        let xx = x * x;
        let mut s = 1.0;
        let mut term = 1.0;
        let mut k = 1.0;
        while k < 60.0 {
            let next = term * (2.0 * k - 1.0) / (2.0 * xx);
            if next.abs() >= term.abs() {
                break;
            }
            s += next;
            term = next;
            k += 1.0;
        }
        xx.exp() / (PI.sqrt() * x) * s
    } else {
        // Series: (2/sqrt(pi)) * sum x^(2k+1)/(k!(2k+1))
        let mut total = 0.0;
        let mut term = x;
        let mut n: u32 = 0;
        loop {
            total += term;
            let nf = f64::from(n + 1);
            term *= x * x * (2.0 * nf - 1.0) / (nf * (2.0 * nf + 1.0));
            n += 1;
            if term.abs() < 1e-18 * total.abs().max(1e-300) || n > 400 {
                break;
            }
        }
        2.0 / PI.sqrt() * (total + term)
    }
}

/// Natural log of the Gamma function.
/// Numerical Recipes gammln (Lanczos g=5, n=6), valid for x > 0.
pub fn log_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // Reflection: Gamma(x)*Gamma(1-x) = pi/sin(pi*x)
        return (PI / (PI * x).sin()).ln() - log_gamma(1.0 - x);
    }

    // NOTE: the series counter must be SEPARATE from the divisor. A previous
    // revision pre-shifted x by -1 and reused the shifted variable both as
    // denominator base and as the final divisor, producing values off by ~ln(x).
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    let mut y = x;
    for c in &COF {
        y += 1.0;
        ser += c / y;
    }
    // Divisor is the ORIGINAL x.
    -tmp + (2.5066282746310005 * ser / x).ln()
}

pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }
    log_gamma(x).exp()
}

pub fn beta(x: f64, y: f64) -> f64 {
    gamma(x) * gamma(y) / gamma(x + y)
}

pub fn log_beta(x: f64, y: f64) -> f64 {
    log_gamma(x) + log_gamma(y) - log_gamma(x + y)
}

pub fn factorial(n: u64) -> f64 {
    if n < 20 {
        (1..=n).fold(1.0, |acc, i| acc * i as f64)
    } else {
        gamma(n as f64 + 1.0)
    }
}

pub fn double_factorial(n: u64) -> f64 {
    if n <= 1 {
        1.0
    } else if n == 2 {
        2.0
    } else {
        n as f64 * double_factorial(n - 2)
    }
}

pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        // Direct rational fit (Hart/NR): J0 is even in x.
        let y = ax * ax;
        let num = 57568490574.0
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0
            + y * (1029532985.0
                + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
        num / den
    } else {
        // Asymptotic form for large |x|.
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let p = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934945152e-7)));
        (0.636619772 / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    }
}

pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        // Direct rational fit; J1 is odd in x.
        let y = ax * ax;
        let mut num = -30.16036606;
        for c in [
            15704.48260,
            -2972611.439,
            242396853.1,
            -7895059235.0,
            72362614232.0,
        ] {
            num = num * y + c;
        }
        let mut den = 1.0;
        for c in [376.9991397, 99447.43394, 18583304.74, 2300535178.0, 144725228442.0] {
            den = den * y + c;
        }
        ax * num / den * x.signum()
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let p = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
        let q = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let r = (0.636619772 / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q);
        if x < 0.0 {
            -r
        } else {
            r
        }
    }
}

pub fn bessel_jn(n: i32, x: f64) -> f64 {
    if n < 0 {
        return (-1.0_f64).powi(n) * bessel_jn(-n, x);
    }
    if x == 0.0 || !x.is_finite() {
        return if n == 0 && x.is_finite() { 1.0 } else { 0.0 };
    }
    match n {
        0 => return bessel_j0(x),
        1 => return bessel_j1(x),
        _ => {}
    }
    let n = n; // silence potential lint
    let neg = x.is_sign_negative();
    let x = x.abs();

    let tox = 2.0 / x;
    let result = if x > f64::from(n) {
        // Upward recurrence is stable for x > n.
        let mut bjm = bessel_j0(x);
        let mut bj = bessel_j1(x);
        for k in 1..n {
            let bjp = tox * f64::from(k) * bj - bjm;
            bjm = bj;
            bj = bjp;
        }
        bj
    } else {
        // Miller's backward algorithm. NOTE: after each update `j_k` holds
        // J_{k-1} — tag/normalize with that shifted index. Normalization uses
        // the identity 1 = J0 + 2*sum(J_{2m}); a previous revision used a
        // J0-ratio that underflowed for x < n and tagged elements off by one.
        let acc = 40.0_f64;
        let top = n + (acc * (n as f64).sqrt()).ceil() as i32 + 8;
        let mut j_kp1 = 0.0_f64; // J_{top+1}
        let mut j_k = 1e-300_f64; // J_top (arbitrary seed)
        let mut ans = 0.0_f64;
        let mut sum = 0.0_f64; // J0 + 2*(J2+J4+...) of swept values
        for k in (1..=top).rev() {
            // before update: j_k holds J_k, j_kp1 holds J_{k+1}
            let j_km1 = tox * f64::from(k) * j_k - j_kp1; // J_{k-1}
            j_kp1 = j_k;
            j_k = j_km1;
            let idx = k - 1;

            if j_k.abs() > 1e100 {
                j_k *= 1e-100;
                j_kp1 *= 1e-100;
                sum *= 1e-100;
                ans *= 1e-100;
            }

            if idx % 2 == 0 {
                sum += j_k * if idx == 0 { 1.0 } else { 2.0 };
            }
            if idx == n {
                ans = j_k;
            }
        }
        ans / sum
    };

    if neg && n % 2 == 1 {
        -result
    } else {
        result
    }
}

pub fn bessel_y0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 8.0 {
        let y = x * x;
        // Rational fit plus the log singular term (a previous revision used
        // sin()/2/(pi*x) here — an unrelated formula entirely).
        let mut num = 228.4622733;
        for c in [
            -86327.92757,
            10879881.29,
            -512359803.6,
            7062834065.0,
            -2957821389.0,
        ] {
            num = num * y + c;
        }
        let mut den = 1.0;
        for c in [226.1030244, 47447.26470, 7189466.438, 745249964.8, 40076544269.0] {
            den = den * y + c;
        }
        num / den + 0.636619772 * bessel_j0(x) * x.ln()
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785398164;
        let p = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * -0.934945152e-7)));
        (0.636619772 / x).sqrt() * (xx.sin() * p + z * xx.cos() * q)
    }
}

pub fn bessel_y1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 8.0 {
        let y = x * x;
        // Direct evaluation with explicit tables (clearest & verified):
        // Canonical NR table (exponents matter — a previous transcription
        // inflated several entries by 10^3..10^6):
        let r = [
            -4.900604943e12,
            1.275274390e12,
            -5.153438139e10,
            7.349264551e8,
            -4.237922726e6,
            8.511937935e3,
        ];
        let sv = [
            2.499580570e13,
            4.244419664e11,
            3.733650367e9,
            2.245904002e7,
            1.020426050e5,
            3.549632885e2,
            1.0,
        ];
        let mut n_val = r[5];
        for i in (0..5).rev() {
            n_val = n_val * y + r[i];
        }
        let mut d_val = sv[6];
        for i in (0..6).rev() {
            d_val = d_val * y + sv[i];
        }
        x * (n_val / d_val) + 0.636619772 * (bessel_j1(x) * x.ln() - 1.0 / x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 2.356194491;
        let p = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
        let q = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        (0.636619772 / x).sqrt() * (xx.sin() * p + z * xx.cos() * q)
    }
}

pub fn bessel_yn(n: i32, x: f64) -> f64 {
    if n == 0 {
        return bessel_y0(x);
    }
    if n == 1 {
        return bessel_y1(x);
    }
    if n < 0 {
        return (-1.0_f64).powi(-n) * bessel_yn(-n, x);
    }

    // Y satisfies the same recurrence as J but MUST be seeded with Y0/Y1;
    // a previous revision seeded with J0/J1, propagating J-values.
    let mut bjm = bessel_y0(x);
    let mut by = bessel_y1(x);

    for j in 1..n {
        let byp = 2.0 * j as f64 / x * by - bjm;
        bjm = by;
        by = byp;
    }

    by
}

pub fn bessel_i0(x: f64) -> f64 {
    if x.abs() < 3.75 {
        let y: f64 = (x / 3.75_f64).powi(2);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
    } else {
        let ax = x.abs();
        (ax.exp() / ax.sqrt())
            * (std::f64::consts::FRAC_2_PI / ax + 0.050001751 + 0.000548 + 0.000042 + 0.000002)
    }
}

pub fn bessel_k0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        let y = x * x / 4.0;
        -y.ln() * bessel_i0(x)
            + (-0.57721566
                + y * (0.42278420
                    + y * (0.23069756 + y * (0.3488590e-1 + y * (0.262698e-2 + y * 0.10750e-3)))))
    } else {
        let y = 2.0 / x;
        (x * (-x).exp()) / x.sqrt()
            * (1.25331414
                + y * (-0.7832358e-1 + y * (0.2189568e-1 + y * (-0.1062446e-1 + y * 0.587872e-2))))
    }
}

/// Spherical Bessel j_n via its own upward-stable recurrence from
/// j0 = sinc(x) and j1 = sin(x)/x^2 - cos(x)/x.
pub fn spherical_jn(n: i32, x: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    if x.abs() < 1e-10 {
        return match n {
            0 => 1.0,
            _ => 0.0,
        };
    }
    match n {
        0 => return x.sin() / x,
        1 => return x.sin() / (x * x) - x.cos() / x,
        _ => {}
    }
    let mut bjm = x.sin() / x;
    let mut bj = x.sin() / (x * x) - x.cos() / x;
    for k in 1..n {
        let bjp = (2.0 * f64::from(k) + 1.0) / x * bj - bjm;
        bjm = bj;
        bj = bjp;
    }
    bj
}

/// Spherical Neumann y_n via upward recurrence from y0, y1.
pub fn spherical_yn(n: i32, x: f64) -> f64 {
    if n < 0 || x <= 0.0 {
        return f64::NAN;
    }
    match n {
        0 => return -x.cos() / x,
        1 => return -x.cos() / (x * x) - x.sin() / x,
        _ => {}
    }
    let mut bjm = -x.cos() / x;
    let mut by = -x.cos() / (x * x) - x.sin() / x;
    for k in 1..n {
        let byp = (2.0 * f64::from(k) + 1.0) / x * by - bjm;
        bjm = by;
        by = byp;
    }
    by
}

pub fn expn(n: i32, x: f64) -> f64 {
    if n < 0 || x < 0.0 {
        return f64::NAN;
    }
    if n == 0 {
        return (-x).exp() / x;
    }
    if x == 0.0 {
        return 1.0 / f64::from(n - 1);
    }

    // Modified-Lentz continued fraction (A&S 5.1.22, Numerical Recipes
    // expint); converges for all x > 0 and every integer order n. A previous
    // revision summed an unrelated power series.
    const MAXIT: usize = 500;
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-300;

    let b_init = x + f64::from(n);
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b_init;
    let mut h = d;

    for i in 1..=MAXIT {
        let an = -(i as f64) * ((n - 1) as f64 + i as f64);
        let b = b_init + 2.0 * i as f64;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    h * (-x).exp()
}

pub fn expi(x: f64) -> f64 {
    if x < 0.0 {
        return -expi(-x);
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }

    if x <= 10.0 {
        // Series: Ei(x) = gamma + ln(x) + sum_{k>=1} x^k/(k*k!)
        // A previous revision dropped the ln(x) term and alternated signs.
        let mut sum = 0.5772156649015329 + x.ln();
        let mut term = 1.0;
        for k in 1..500u32 {
            term *= x / f64::from(k);
            let add = term / f64::from(k);
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        sum
    } else {
        // Asymptotic: e^x/x * sum k!/x^k, truncated at the smallest term.
        // The old branch used e^{-x} and overwrote the accumulator.
        let mut acc = 1.0;
        let mut t = 1.0;
        let mut k: u32 = 1;
        loop {
            let next = t * f64::from(k) / x;
            if next.abs() >= t.abs() || k > 200 {
                break;
            }
            acc += next;
            t = next;
            k += 1;
        }
        x.exp() / x * acc
    }
}

#[cfg(test)]
mod numeric_reference_tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn test_gamma_reference_values() {
        // Reference values from scipy.special.gamma
        assert!(close(gamma(0.5), 1.7724538509055159, 1e-12));
        assert!(close(gamma(1.0), 1.0, 1e-14));
        assert!(close(gamma(2.0), 1.0, 1e-13));
        assert!(close(gamma(5.0), 24.0, 1e-12));
        assert!(close(gamma(10.0), 362880.0, 1e-11));
    }

    #[test]
    fn test_log_gamma_reference_values() {
        // scipy.special.gammaln
        assert!(close(log_gamma(0.5), 0.57236494292470008, 1e-12));
        assert!(close(log_gamma(1.0), 0.0, 1e-14));
        assert!(close(log_gamma(3.0), 0.69314718055994529, 1e-12));
        assert!(close(log_gamma(10.0), 12.801827480081469, 1e-12));
    }

    #[test]
    fn test_factorial_large() {
        // Regression: broken gamma made factorial(20) = 41.10
        assert!(close(factorial(20), 2_432_902_008_176_640_000.0, 1e-8));
        assert!(factorial(171).is_infinite(), "171! overflows f64");
    }

    #[test]
    fn test_erfi_reference_values() {
        // scipy.special.erfi
        assert!(close(erfi(0.5), 0.61495209469651098, 1e-9));
        assert!(close(erfi(2.0), 18.5648024146958, 1e-9));
        assert!(close(erfi(5.0), 8.29827388756904e9, 1e-9));
    }

    #[test]
    fn test_bessel_jn_reference_values() {
        // scipy.special.jv
        assert!(close(bessel_jn(2, 1.0), 0.11490348493190048, 1e-9));
        assert!(close(bessel_jn(2, 5.0), 0.04656511627775229, 1e-7));
        assert!(close(bessel_jn(5, 2.0), 0.0070396297558716855, 1e-9));
        assert!(close(bessel_jn(10, 15.0), -0.090071811047659034, 1e-9));
    }

    #[test]
    fn test_bessel_yn_reference_values() {
        // scipy.special.yv — regression: seeded with J instead of Y
        assert!(close(bessel_yn(2, 1.0), -1.6506826068162554, 1e-7));
        assert!(close(bessel_yn(3, 1.0), -5.8215176059696515, 1e-7));
    }

    #[test]
    fn test_spherical_bessel_reference_values() {
        // scipy.special.spherical_jn / spherical_yn — regression: previously
        // aliased to ordinary Bessel functions.
        assert!(close(spherical_jn(0, 1.0), 0.84147098480789650, 1e-12));
        assert!(close(spherical_jn(1, 1.0), 0.30116867893975674, 1e-12));
        assert!(close(spherical_yn(0, 1.0), -0.54030230586813977, 1e-12));
        assert!(close(spherical_yn(1, 1.0), -1.3817732906760399, 1e-12));
    }

    #[test]
    fn test_expn_reference_values() {
        // scipy.special.expn — regression: computed an unrelated series.
        assert!(close(expn(1, 1.0), 0.21938393439552027, 1e-9));
        assert!(close(expn(2, 1.0), 0.14849550677592205, 1e-9));
        assert!(close(expn(3, 0.5), 0.22160436418326517, 1e-9));
        assert!(close(expn(4, 12.0), 3.89586007095526e-7, 1e-6));
        assert!(close(expn(3, 25.0), 4.97790974813523e-13, 1e-6));
    }

    #[test]
    fn test_expi_reference_values() {
        // scipy.special.expi — regression: dropped ln(x), alternated signs,
        // and used e^{-x} in the asymptotic branch.
        assert!(close(expi(1.0), 1.8951178163559368, 1e-9));
        assert!(close(expi(5.0), 40.18527535580318, 1e-9));
        assert!(close(expi(20.0), 25615652.664056595, 1e-7));
        assert_eq!(expi(0.0), f64::NEG_INFINITY);
    }
}
#[cfg(test)]
mod bessel_debug {
    use super::*;
    #[test]
    fn probe() {
        eprintln!("jn(2,1)={}", bessel_jn(2, 1.0));
        eprintln!("jn(2,5)={}", bessel_jn(2, 5.0));
        eprintln!("j0(5)={} j1(5)={}", bessel_j0(5.0), bessel_j1(5.0));
        eprintln!("yn(2,1)={}", bessel_yn(2, 1.0));
        eprintln!("y0(1)={} y1(1)={}", bessel_y0(1.0), bessel_y1(1.0));
    }
}

