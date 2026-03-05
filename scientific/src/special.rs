use std::f64::consts::{E, PI};

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
    if x.abs() < 1.0 {
        let mut sum = 0.0;
        let mut term = x;
        for n in 0..100 {
            sum += term;
            term *= x * x / ((2 * n + 3) as f64);
        }
        2.0 / PI.sqrt() * sum
    } else {
        let mut sum = 0.0;
        let mut term = 1.0 / x;
        for n in 0..100 {
            sum += term;
            term *= (2 * n + 1) as f64 / (x * x);
        }
        (E * x).exp() / (PI * x).sqrt() * sum
    }
}

pub fn gamma(x: f64) -> f64 {
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }

    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let c = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];

        let mut y = x;
        let mut tmp = x + 5.5;
        tmp -= (x + 0.5) * tmp.ln();
        let mut ser = 1.000000000190015;

        for c_i in &c {
            y += 1.0;
            ser += c_i / y;
        }

        -tmp + (2.5066282746310005 * ser / x).ln().exp()
    }
}

pub fn log_gamma(x: f64) -> f64 {
    gamma(x).ln()
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
    if x.abs() < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0
            + y * (-13362599354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0
            + y * (1029532985.0
                + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));

        if x.abs() < 1e-10 {
            1.0
        } else {
            ans1 / ans2
        }
    } else {
        let ax = x.abs();
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;

        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));

        (2.0 / ax).sqrt() * (xx.cos() * ans1 + z * xx.sin() * ans2)
    }
}

pub fn bessel_j1(x: f64) -> f64 {
    if x.abs() < 8.0 {
        let y = x * x;
        let ans1 = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0
            + y * (2300535178.0
                + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));

        if x.abs() < 1e-10 {
            0.0
        } else {
            ans1 / ans2
        }
    } else {
        let ax = x.abs();
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;

        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));

        let result = (2.0 / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 {
            -result
        } else {
            result
        }
    }
}

pub fn bessel_jn(n: i32, x: f64) -> f64 {
    if n == 0 {
        return bessel_j0(x);
    }
    if n == 1 {
        return bessel_j1(x);
    }
    if n < 0 {
        return (-1.0_f64).powi(-n) * bessel_jn(-n, x);
    }

    let mut bjp = 0.0;
    let mut bj = 1.0;
    let mut sum = 0.0;

    let m = 2 * ((n as f64 + (40.0 * x.abs()).sqrt()) as usize);

    for j in (1..=m).rev() {
        let bjm = 2.0 * (j as f64) / x * bj - bjp;
        bjp = bj;
        bj = bjm;

        if bj.abs() > 1e10 {
            bj *= 1e-10;
            bjp *= 1e-10;
            sum *= 1e-10;
        }

        if j as i32 == n {
            sum = bjp;
        }
    }

    sum * bessel_j0(x) / bjp
}

pub fn bessel_y0(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 8.0 {
        let y = x * x;
        let ans1 = -2957821389.0
            + y * (7062834060.0
                + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
        let ans2 = 40076544269.0
            + y * (745249964.8
                + y * (7189466.438 + y * (47447.26470 + y * (883.8835063 + y * 1.0))));

        (ans1 / ans2) * x.sin() - 2.0 / (PI * x)
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 0.785398164;

        let ans1 = 1.0
            + y * (-0.1098628627e-2
                + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1
            + y * (0.1430488765e-3
                + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));

        (2.0 / x).sqrt() * (xx.sin() * ans1 + z * xx.cos() * ans2)
    }
}

pub fn bessel_y1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 8.0 {
        let y = x * x;
        let ans1 = x
            * (-0.4900602113e13
                + y * (0.1275273940e13
                    + y * (-0.5153438139e11
                        + y * (0.7349265431e9 + y * (-0.4237922726e7 + y * 85126.467717)))));
        let ans2 = 0.2499580570e14
            + y * (0.4244419664e12
                + y * (0.3733650367e10
                    + y * (0.2245904002e8 + y * (0.1020428050e6 + y * (0.3549632885e3 + y)))));

        (ans1 / ans2) * x.sin() - 2.0 / (PI * x) - x.cos() * (2.0 / (PI * x * x))
    } else {
        let z = 8.0 / x;
        let y = z * z;
        let xx = x - 2.356194491;

        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));

        (2.0 / x).sqrt() * (xx.sin() * ans1 - z * xx.cos() * ans2)
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

    let mut bjp = bessel_j1(x);
    let mut bj = bessel_j0(x);
    let mut result = bessel_y1(x);

    for j in 1..n {
        let bjm = 2.0 * j as f64 / x * bj - bjp;
        bjp = bj;
        bj = bjm;
        result = bj;
    }

    result
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
        (ax.exp() / ax.sqrt()) * (std::f64::consts::FRAC_2_PI / ax + 0.050001751 + 0.000548 + 0.000042 + 0.000002)
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

pub fn spherical_jn(n: i32, x: f64) -> f64 {
    bessel_jn(n, x)
}

pub fn spherical_yn(n: i32, x: f64) -> f64 {
    bessel_yn(n, x)
}

pub fn expn(n: i32, x: f64) -> f64 {
    if n <= 0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0 / (n - 1) as f64;
    }

    let mut sum = 1.0 / n as f64;
    let mut term = 1.0 / n as f64;

    for k in 1..100 {
        term *= -x / (n as f64 + k as f64);
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
    }

    sum
}

pub fn expi(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 1.0 {
        let mut sum = 0.0;
        let mut term = x;
        for n in 1..100 {
            sum += term / n as f64;
            term *= -x;
            if term.abs() < 1e-15 {
                break;
            }
        }
        sum + 0.5772156649
    } else {
        let mut sum = 0.0;
        let mut term = 1.0;
        for n in 1..100 {
            term *= n as f64 / x;
            sum += term;
            if term < 1e-15 {
                break;
            }
        }
        (-x).exp() / x * (1.0 + sum)
    }
}
