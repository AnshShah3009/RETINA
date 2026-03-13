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
    if data.is_empty() {
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
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    if x.len() < 2 {
        return Err("Need at least 2 data points".to_string());
    }
    let mx = mean(x);
    let my = mean(y);
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom < 1e-30 {
        return Err("Zero variance in one or both inputs".to_string());
    }
    Ok(sxy / denom)
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
pub fn t_test_1sample(data: &[f64], mu: f64) -> (f64, f64) {
    let n = data.len() as f64;
    let m = mean(data);
    let s = sample_std(data);
    if s < 1e-30 || n < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let t = (m - mu) / (s / n.sqrt());
    let df = n - 1.0;
    let p = 2.0 * (1.0 - t_cdf(t.abs(), df));
    (t, p)
}

/// Two-sample t-test (Welch's, unequal variances).
///
/// Returns `(t_statistic, p_value)` for a two-sided test of H0: mean_a = mean_b.
pub fn t_test_2sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    if na < 2.0 || nb < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let ma = mean(a);
    let mb = mean(b);
    let va = sample_var(a);
    let vb = sample_var(b);

    let se = (va / na + vb / nb).sqrt();
    if se < 1e-30 {
        return (f64::NAN, f64::NAN);
    }
    let t = (ma - mb) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (va / na + vb / nb).powi(2);
    let den = (va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0);
    let df = num / den;

    let p = 2.0 * (1.0 - t_cdf(t.abs(), df));
    (t, p)
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

/// CDF of Student's t-distribution via regularized incomplete beta function.
fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 + 0.5 * (1.0 - regularized_incomplete_beta(df / 2.0, 0.5, x)) * t.signum()
}

/// Regularized incomplete beta function I_x(a,b) using continued fraction (Lentz's method).
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Use symmetry relation if x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();

    // Continued fraction (Lentz's method)
    let tiny = 1e-30;
    let eps = 1e-14;
    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(tiny);
    let mut h = d;

    for m in 1..=200 {
        let m_f = m as f64;

        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 / (1.0 + num_even * d).max(tiny);
        c = (1.0 + num_even / c).max(tiny);
        h *= d * c;

        // Odd step
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 / (1.0 + num_odd * d).max(tiny);
        c = (1.0 + num_odd / c).max(tiny);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    (ln_prefix.exp() * h).clamp(0.0, 1.0)
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let x = x - 1.0;
    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for c in &coeffs {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / (x + 1.0)).ln()
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

    /// Cumulative distribution function (uses erf from the special module).
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / (self.std * std::f64::consts::SQRT_2);
        0.5 * (1.0 + crate::special::erf(z))
    }

    /// Percent point function (inverse CDF) using rational approximation.
    ///
    /// Abramowitz and Stegun approximation 26.2.23.
    pub fn ppf(&self, p: f64) -> f64 {
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

/// Inverse standard normal CDF (Beasley-Springer-Moro algorithm).
fn inv_normal_cdf(p: f64) -> f64 {
    // Rational approximation for the central region
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

    if ss_xx < 1e-30 {
        return Err("Zero variance in x".to_string());
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

    let r_squared = if ss_yy > 1e-30 {
        1.0 - ss_res / ss_yy
    } else {
        1.0
    };

    let mse = ss_res / (n - 2) as f64;
    let std_err = (mse / ss_xx).sqrt();

    // t-statistic for slope
    let t = if std_err > 1e-30 {
        slope / std_err
    } else {
        f64::INFINITY
    };
    let df = (n - 2) as f64;
    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));

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
/// and `counts` has length `bins`.
pub fn histogram(data: &[f64], bins: usize) -> (Vec<f64>, Vec<usize>) {
    if data.is_empty() || bins == 0 {
        return (Vec::new(), Vec::new());
    }

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let range = if (max - min).abs() < 1e-30 {
        1.0
    } else {
        max - min
    };

    let bin_width = range / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();
    let mut counts = vec![0usize; bins];

    for &v in data {
        let mut idx = ((v - min) / bin_width) as usize;
        if idx >= bins {
            idx = bins - 1;
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
