//! Statistical functions.
//!
//! Provides descriptive statistics, correlation, hypothesis testing,
//! distributions, linear regression, and histogram computation.

use std::f64::consts::PI;

/// Arithmetic mean. Returns 0.0 for empty input.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Median (returns NaN for empty input).
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    // NaN previously panicked the sort; NaN propagates instead (matches f64
    // semantics elsewhere in this crate).
    if data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Population variance.
pub fn variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Population standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Percentile (p in [0, 100]). Uses linear interpolation between nearest ranks.
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() || p.is_nan() {
        return f64::NAN;
    }
    if data.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();

    let p = p.clamp(0.0, 100.0);
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Pearson correlation coefficient.
///
/// Deviations are scaled by their largest magnitude before squaring, so data
/// of any magnitude (1e-200 or 1e200) neither underflows nor overflows; the
/// zero-variance test is relative to the data, not an absolute `1e-30`.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    if x.len() < 2 {
        return Err("Need at least 2 data points".to_string());
    }
    let mx = mean(x);
    let my = mean(y);
    let (Some(kx), Some(ky)) = (deviation_scale(x, mx), deviation_scale(y, my)) else {
        return Err("Zero variance in one or both inputs".to_string());
    };
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = (xi - mx) / kx;
        let dy = (yi - my) / ky;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    Ok((sxy / (sxx.sqrt() * syy.sqrt())).clamp(-1.0, 1.0))
}

/// Largest `|v - m|`, or `None` if the data are constant to rounding: every
/// deviation is within a few ulps of the data's magnitude, which is all the
/// error that computing the mean `m` can introduce.
fn deviation_scale(data: &[f64], m: f64) -> Option<f64> {
    let max_dev = data.iter().fold(0.0f64, |a, &v| a.max((v - m).abs()));
    let max_abs = data.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    if max_dev.is_nan() || max_dev <= 4.0 * f64::EPSILON * max_abs || max_dev == 0.0 {
        None
    } else {
        Some(max_dev)
    }
}

/// Sample covariance between x and y.
pub fn covariance(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    if x.len() < 2 {
        return Err("Need at least 2 data points".to_string());
    }
    let mx = mean(x);
    let my = mean(y);
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum::<f64>()
        / (x.len() - 1) as f64;
    Ok(cov)
}

/// Covariance matrix for multivariate data.
///
/// `data` is a slice of variables, each a `Vec<f64>` of the same length.
/// Returns a `p x p` matrix (row-major, as `Vec<Vec<f64>>`) where `p = data.len()`.
pub fn covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let p = data.len();
    if p == 0 {
        return Vec::new();
    }
    let n = data[0].len();
    // Ragged input previously indexed out of bounds and panicked.
    if data.iter().any(|col| col.len() != n) {
        panic!("covariance_matrix: all variables must have equal length");
    }
    if n < 2 {
        panic!("covariance_matrix: need at least 2 observations");
    }
    let means: Vec<f64> = data.iter().map(|v| mean(v)).collect();

    let mut cov = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in i..p {
            let c: f64 = (0..n)
                .map(|k| (data[i][k] - means[i]) * (data[j][k] - means[j]))
                .sum::<f64>()
                / if n > 1 { (n - 1) as f64 } else { 1.0 };
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    cov
}

/// One-sample t-test.
///
/// Returns `(t_statistic, p_value)` for a two-sided test of H0: mean = mu.
/// The p-value is the t-distribution's two-sided tail computed directly
/// (regularized incomplete beta), not `1 - cdf`, so it stays accurate far
/// below 1e-16. Constant data (to rounding) give `(NaN, NaN)`.
pub fn t_test_1sample(data: &[f64], mu: f64) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let m = mean(data);
    if deviation_scale(data, m).is_none() {
        return (f64::NAN, f64::NAN);
    }
    let s = sample_std(data);
    let t = (m - mu) / (s / n.sqrt());
    let df = n - 1.0;
    (t, t_two_sided_p(t, df))
}

/// Two-sample t-test (Welch's, unequal variances).
///
/// Returns `(t_statistic, p_value)` for a two-sided test of H0: mean_a = mean_b.
/// Two constant samples (to rounding) give `(NaN, NaN)`.
pub fn t_test_2sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    if na < 2.0 || nb < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let ma = mean(a);
    let mb = mean(b);
    if deviation_scale(a, ma).is_none() && deviation_scale(b, mb).is_none() {
        return (f64::NAN, f64::NAN);
    }
    let va = sample_var(a);
    let vb = sample_var(b);

    let se = (va / na + vb / nb).sqrt();
    let t = (ma - mb) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (va / na + vb / nb).powi(2);
    let den = (va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0);
    let df = num / den;

    (t, t_two_sided_p(t, df))
}

// Sample variance (with Bessel's correction)
fn sample_var(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

fn sample_std(data: &[f64]) -> f64 {
    sample_var(data).sqrt()
}

/// Two-sided tail `P(|T| >= |t|)` of Student's t with `df` degrees of freedom:
/// `I_{df/(df+t²)}(df/2, 1/2)`, with both `x` and `1 - x` formed without
/// cancellation. Replaces `2 * (1 - cdf(|t|))`, which was exactly 0 for every
/// p-value below ~1.1e-16.
fn t_two_sided_p(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    let t2 = t * t;
    let denom = df + t2;
    regularized_incomplete_beta(df / 2.0, 0.5, df / denom, t2 / denom)
}

/// Regularized incomplete beta function `I_x(a, b)`, with `y = 1 - x` supplied
/// by the caller so it carries no cancellation. Continued fraction (modified
/// Lentz, Numerical Recipes `betacf`) on whichever side converges.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if y == 0.0 {
        return 1.0;
    }

    // Symmetry I_x(a, b) = 1 - I_{1-x}(b, a) where the CF converges faster.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, y, x);
    }

    let ln_prefix = a * x.ln() + b * y.ln() - ln_beta(a, b) - a.ln();

    const TINY: f64 = 1e-300;
    let guard = |v: f64| if v.abs() < TINY { TINY } else { v };
    let mut c = 1.0;
    let mut d = 1.0 / guard(1.0 - (a + b) * x / (a + 1.0));
    let mut h = d;

    for m in 1..=300 {
        let m_f = m as f64;

        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / guard(1.0 + num_even * d);
        c = guard(1.0 + num_even / c);
        h *= d * c;

        // Odd step
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / guard(1.0 + num_odd * d);
        c = guard(1.0 + num_odd / c);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }

    (ln_prefix.exp() * h).clamp(0.0, 1.0)
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn ln_gamma(x: f64) -> f64 {
    // Delegate to the corrected Numerical Recipes implementation. The
    // previous local copy shifted x by -1 AND divided by (x+1), double-
    // shifting the Lanczos denominators and biasing every downstream
    // t-test p-value (e.g. t_cdf(2, df=4): 0.977 vs correct 0.942).
    crate::special::log_gamma(x)
}

/// Normal distribution.
pub struct NormalDistribution {
    pub mean: f64,
    pub std: f64,
}

impl NormalDistribution {
    /// Create a new normal distribution with the given mean and standard deviation.
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    /// Probability density function.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std;
        (-0.5 * z * z).exp() / (self.std * (2.0 * PI).sqrt())
    }

    /// Cumulative distribution function, `erfc(-z / √2) / 2`.
    ///
    /// Through `erfc`, so the lower tail keeps full relative accuracy:
    /// `0.5 * (1 + erf(z))` cancelled to exactly 0 for `z <~ -8.3`
    /// (`cdf(-10)` returned 0; SciPy `7.61985302416047e-24`).
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / (self.std * std::f64::consts::SQRT_2);
        0.5 * crate::special::erfc(-z)
    }

    /// Survival function `1 - cdf(x)`, computed directly as `erfc(z / √2) / 2`
    /// so the upper tail does not cancel (`sf(10) = 7.6e-24`, not 0).
    pub fn sf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / (self.std * std::f64::consts::SQRT_2);
        0.5 * crate::special::erfc(z)
    }

    /// Percent point function (inverse CDF).
    ///
    /// Acklam's rational approximation (relative error ~1e-9) as the starting
    /// point, then one Halley step on the `erfc`-based cdf, which brings it to
    /// machine precision across `[1e-300, 1 - 1e-16]` (matches
    /// `scipy.stats.norm.ppf` to a few ulps). The previous version stopped at
    /// the raw approximation and was documented as A&S 26.2.23, which it is
    /// not. `ppf(0) = -inf`, `ppf(1) = +inf`, NaN in gives NaN.
    pub fn ppf(&self, p: f64) -> f64 {
        if p.is_nan() {
            return f64::NAN;
        }
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        let z = inv_normal_cdf(p);
        self.mean + z * self.std
    }
}

/// Inverse standard normal CDF: Acklam's approximation refined by Halley.
fn inv_normal_cdf(p: f64) -> f64 {
    if p > 0.5 {
        // 1 - p is exact for p >= 0.5 (Sterbenz), and refining in the lower
        // tail keeps the residual cdf(z) - q free of cancellation.
        return -inv_normal_cdf_lower(1.0 - p);
    }
    inv_normal_cdf_lower(p)
}

/// `p <= 0.5`.
fn inv_normal_cdf_lower(p: f64) -> f64 {
    if p == 0.5 {
        return 0.0;
    }
    let z = acklam(p);
    // Halley step on F(z) = Φ(z) - p, with Φ via erfc: z <= 0 here, so Φ(z)
    // is a lower-tail value with full relative accuracy.
    let e = 0.5 * crate::special::erfc(-z / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * PI).sqrt() * (0.5 * z * z).exp();
    if !u.is_finite() {
        return z;
    }
    z - u / (1.0 + 0.5 * z * u)
}

/// Peter Acklam's rational approximation to the inverse normal CDF
/// (relative error below 1.15e-9).
fn acklam(p: f64) -> f64 {
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Simple linear regression: y = slope * x + intercept.
///
/// Returns a `LinregressResult` with slope, intercept, R-squared, standard error, and p-value.
/// The p-value (two-sided, H0: slope = 0) is the t-distribution tail computed
/// directly, so it is not rounded to 0 below ~1e-16 (SciPy-equivalent; on a
/// near-perfect 30-point line: 1.44e-96, where `1 - cdf` gave 0). `x` that is
/// constant to rounding is an error; constant `y` gives `r_squared = 1`.
pub fn linregress(x: &[f64], y: &[f64]) -> Result<LinregressResult, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    let n = x.len();
    if n < 3 {
        return Err("Need at least 3 data points for regression".to_string());
    }

    let mx = mean(x);
    let my = mean(y);
    if deviation_scale(x, mx).is_none() {
        return Err("Zero variance in x".to_string());
    }
    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    let mut ss_yy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
        ss_yy += dy * dy;
    }

    let slope = ss_xy / ss_xx;
    let intercept = my - slope * mx;

    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            let pred = slope * xi + intercept;
            (yi - pred).powi(2)
        })
        .sum();

    let r_squared = if deviation_scale(y, my).is_some() {
        1.0 - ss_res / ss_yy
    } else {
        1.0
    };

    let mse = ss_res / (n - 2) as f64;
    let std_err = (mse / ss_xx).sqrt();

    // t-statistic for slope; an exact fit has std_err == 0 and t = ±inf.
    let t = if std_err > 0.0 {
        slope / std_err
    } else if slope == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let df = (n - 2) as f64;
    let p_value = t_two_sided_p(t, df);

    Ok(LinregressResult {
        slope,
        intercept,
        r_squared,
        std_err,
        p_value,
    })
}

/// Result of a linear regression.
pub struct LinregressResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub std_err: f64,
    pub p_value: f64,
}

/// Compute a histogram of the data.
///
/// Returns `(bin_edges, counts)` where `bin_edges` has length `bins + 1`
/// and `counts` has length `bins`, like `numpy.histogram(data, bins)`: equal
/// bins over `[min, max]`, each half-open except the last, which includes
/// `max`; constant data get the range `[v - 0.5, v + 0.5]`.
///
/// Non-finite values (NaN, ±inf) are not counted - NaN used to land in bin 0
/// because `NaN as usize` is 0. With no finite values the result is empty.
/// (numpy raises for NaN/inf with an automatic range.) The constant-data test
/// is exact; it was an absolute `1e-30`, which treated `[1e-40, 3e-40]` as a
/// single value.
pub fn histogram(data: &[f64], bins: usize) -> (Vec<f64>, Vec<usize>) {
    let finite = data.iter().copied().filter(|v| v.is_finite());
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in finite.clone() {
        min = min.min(v);
        max = max.max(v);
    }
    if bins == 0 || min > max {
        return (Vec::new(), Vec::new());
    }
    if min == max {
        min -= 0.5;
        max += 0.5;
    }

    let range = max - min;
    let bin_width = range / bins as f64;
    let edges: Vec<f64> = (0..=bins)
        .map(|i| {
            if i == bins {
                max
            } else {
                min + i as f64 * bin_width
            }
        })
        .collect();
    let mut counts = vec![0usize; bins];

    // numpy's binning: scaled index, then corrected against the actual
    // edges so a value equal to an edge always lands in the bin it opens.
    let norm = bins as f64 / range;
    for v in finite {
        let mut idx = (((v - min) * norm) as usize).min(bins - 1);
        if v < edges[idx] {
            idx -= 1;
        } else if idx + 1 < bins && v >= edges[idx + 1] {
            idx += 1;
        }
        counts[idx] += 1;
    }

    (edges, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_and_std() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&data) - 5.0).abs() < 1e-10);
        assert!((std_dev(&data) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median() {
        assert!((median(&[1.0, 3.0, 5.0]) - 3.0).abs() < 1e-10);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // perfect positive correlation
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);

        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // perfect negative
        let r_neg = pearson_correlation(&x, &y_neg).unwrap();
        assert!((r_neg - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_t_test_1sample() {
        // Data with mean ~5.0, test against mu=5
        let data = vec![4.5, 5.1, 4.8, 5.3, 5.0, 4.9, 5.2, 4.7];
        let (t, p) = t_test_1sample(&data, 5.0);
        // t should be close to 0, p should be large (fail to reject)
        assert!(t.abs() < 2.0, "t-statistic too large: {}", t);
        assert!(p > 0.05, "p-value too small: {}", p);
    }

    #[test]
    fn test_t_test_2sample() {
        // Two clearly different groups
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let (_, p) = t_test_2sample(&a, &b);
        assert!(p < 0.01, "Should reject H0, p={}", p);
    }

    #[test]
    fn test_normal_distribution() {
        let n = NormalDistribution::new(0.0, 1.0);
        // PDF at 0 should be 1/sqrt(2*pi) ~ 0.3989
        assert!((n.pdf(0.0) - 0.3989422804).abs() < 1e-6);
        // CDF at 0 should be 0.5
        assert!((n.cdf(0.0) - 0.5).abs() < 1e-6);
        // CDF at large positive should be ~1
        assert!((n.cdf(5.0) - 1.0).abs() < 1e-4);
        // PPF(0.5) should be 0
        assert!(n.ppf(0.5).abs() < 1e-4);
        // PPF(CDF(1.0)) should be ~1.0
        assert!((n.ppf(n.cdf(1.0)) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_linregress() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1]; // ~y = 2x + 0
        let result = linregress(&x, &y).unwrap();
        assert!((result.slope - 2.0).abs() < 0.1, "slope={}", result.slope);
        assert!(result.r_squared > 0.99, "r^2={}", result.r_squared);
        assert!(result.p_value < 0.01, "p={}", result.p_value);
    }

    #[test]
    fn test_histogram() {
        let data = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let (edges, counts) = histogram(&data, 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.len(), 5);
        let total: usize = counts.iter().sum();
        assert_eq!(total, data.len());
    }

    #[test]
    fn test_covariance_matrix() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let cov = covariance_matrix(&[x, y]);
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        // Var(x) = 1, Cov(x,y) = 2, Var(y) = 4
        assert!((cov[0][0] - 1.0).abs() < 1e-10);
        assert!((cov[0][1] - 2.0).abs() < 1e-10);
        assert!((cov[1][1] - 4.0).abs() < 1e-10);
    }
}
