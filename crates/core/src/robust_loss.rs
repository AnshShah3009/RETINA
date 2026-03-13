//! Robust loss functions for outlier-aware optimization
//!
//! This module provides robust loss functions commonly used in computer vision
//! for optimization problems with outliers. The functions support Graduated
//! Non-Convexity (GNC) optimization via parameter adjustment.

/// Robust loss functions for optimization
#[derive(Debug, Clone, Copy)]
pub enum RobustLoss {
    /// Geman-McClure: ρ(r) = (μ * r²) / (μ + r²)
    GemanMcClure { mu: f32 },
    /// Welsch/Leclerc: ρ(r) = μ * (1 - exp(-r²/μ))
    Welsch { mu: f32 },
    /// Huber: ρ(r) = { 0.5*r² if |r|≤μ, μ*(|r| - 0.5*μ) otherwise }
    Huber { mu: f32 },
    /// Truncated Least Squares (TLS): ρ(r) = { r² if r² < c², c² otherwise }
    TruncatedLeastSquares { c: f32 },
    /// Cauchy: ρ(r) = (μ²/2) * log(1 + (r/μ)²)
    Cauchy { mu: f32 },
    /// Tukey/biweight: ρ(r) = { (μ²/6)*(1 - (1-(r/μ)²)³) if |r|≤μ, μ²/6 otherwise }
    Tukey { mu: f32 },
}

impl RobustLoss {
    /// Evaluate the loss function ρ(r)
    pub fn evaluate(&self, residual: f32) -> f32 {
        let r = residual.abs();
        match self {
            RobustLoss::GemanMcClure { mu } => (mu * r * r) / (mu + r * r),
            RobustLoss::Welsch { mu } => mu * (1.0 - (-r * r / mu).exp()),
            RobustLoss::Huber { mu } => {
                if r <= *mu {
                    0.5 * r * r
                } else {
                    mu * (r - 0.5 * mu)
                }
            }
            RobustLoss::TruncatedLeastSquares { c } => {
                if r < *c {
                    r * r
                } else {
                    c * c
                }
            }
            RobustLoss::Cauchy { mu } => (mu * mu / 2.0) * (1.0 + (r / mu).powi(2)).ln(),
            RobustLoss::Tukey { mu } => {
                if r <= *mu {
                    let t = 1.0 - (r / mu).powi(2);
                    (mu * mu / 6.0) * (1.0 - t.powi(3))
                } else {
                    mu * mu / 6.0
                }
            }
        }
    }

    /// Compute weight for reweighted least squares
    /// weight = ρ'(r) / r (or 1 if r ≈ 0)
    pub fn weight(&self, residual: f32) -> f32 {
        let r = residual.abs();
        if r < 1e-6 {
            return 1.0;
        }

        match self {
            RobustLoss::GemanMcClure { mu } => {
                let r2 = r * r;
                (2.0 * mu * mu) / ((mu + r2) * (mu + r2))
            }
            RobustLoss::Welsch { mu } => (-r * r / mu).exp(),
            RobustLoss::Huber { mu } => {
                if r <= *mu {
                    1.0
                } else {
                    mu / r
                }
            }
            RobustLoss::TruncatedLeastSquares { c } => {
                if r < *c {
                    1.0
                } else {
                    0.0 // Zero weight for outliers
                }
            }
            RobustLoss::Cauchy { mu } => 1.0 / (1.0 + (r / mu).powi(2)),
            RobustLoss::Tukey { mu } => {
                if r <= *mu {
                    let t = 1.0 - (r / mu).powi(2);
                    t * t
                } else {
                    0.0
                }
            }
        }
    }

    /// Get the current robust loss parameter
    pub fn get_param(&self) -> f32 {
        match self {
            RobustLoss::GemanMcClure { mu } => *mu,
            RobustLoss::Welsch { mu } => *mu,
            RobustLoss::Huber { mu } => *mu,
            RobustLoss::TruncatedLeastSquares { c } => *c,
            RobustLoss::Cauchy { mu } => *mu,
            RobustLoss::Tukey { mu } => *mu,
        }
    }

    /// Update the parameter for GNC optimization
    pub fn update_param(&mut self, new_param: f32) {
        match self {
            RobustLoss::GemanMcClure { mu } => *mu = new_param,
            RobustLoss::Welsch { mu } => *mu = new_param,
            RobustLoss::Huber { mu } => *mu = new_param,
            RobustLoss::TruncatedLeastSquares { c } => *c = new_param,
            RobustLoss::Cauchy { mu } => *mu = new_param,
            RobustLoss::Tukey { mu } => *mu = new_param,
        }
    }

    /// Compute weight using direct error magnitude (for pose graph and similar uses)
    /// This is equivalent to weight() but accepts the error magnitude directly
    pub fn compute_weight(&self, error: f32) -> f32 {
        self.weight(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geman_mcclure() {
        let loss = RobustLoss::GemanMcClure { mu: 1.0 };
        assert!(loss.evaluate(0.0) == 0.0);
        assert!(loss.evaluate(1.0) == 0.5); // (1 * 1²) / (1 + 1²) = 0.5
        assert!(loss.evaluate(2.0) > 0.5); // Increases with error
        assert!(loss.weight(0.0) == 1.0); // Near zero returns 1.0
        assert!(loss.weight(2.0) < 1.0); // Weight decreases with error
    }

    #[test]
    fn test_huber() {
        let loss = RobustLoss::Huber { mu: 1.0 };
        assert!(loss.evaluate(0.0) == 0.0);
        assert!(loss.weight(0.5) == 1.0); // Within threshold
        assert!(loss.weight(2.0) < 1.0); // Outside threshold
    }

    #[test]
    fn test_tukey() {
        let loss = RobustLoss::Tukey { mu: 1.0 };
        assert!(loss.evaluate(0.0) == 0.0);
        assert!(loss.weight(0.5) < 1.0); // Tukey has lower weights
        assert!(loss.weight(1.1) == 0.0); // Beyond threshold is zero
    }

    #[test]
    fn test_welsch() {
        let loss = RobustLoss::Welsch { mu: 1.0 };
        assert!(loss.evaluate(0.0) == 0.0);
        let w1 = loss.weight(0.5);
        let w2 = loss.weight(1.5);
        assert!(w1 > w2); // Weight decreases with error
    }

    #[test]
    fn test_cauchy() {
        let loss = RobustLoss::Cauchy { mu: 1.0 };
        assert!(loss.evaluate(0.0) == 0.0);
        assert!(loss.weight(0.0) == 1.0);
        assert!(loss.weight(2.0) < 1.0);
    }

    #[test]
    fn test_truncated_least_squares() {
        let loss = RobustLoss::TruncatedLeastSquares { c: 1.0 };
        assert!(loss.evaluate(0.0) == 0.0);
        assert!(loss.evaluate(0.5) == 0.25);
        assert!(loss.evaluate(1.5) == 1.0); // Saturated
        assert!(loss.weight(0.5) == 1.0); // Within threshold
        assert!(loss.weight(1.1) == 0.0); // Beyond threshold
    }

    #[test]
    fn test_get_param() {
        let loss = RobustLoss::GemanMcClure { mu: 2.5 };
        assert_eq!(loss.get_param(), 2.5);

        let loss = RobustLoss::TruncatedLeastSquares { c: 1.5 };
        assert_eq!(loss.get_param(), 1.5);
    }

    #[test]
    fn test_update_param() {
        let mut loss = RobustLoss::Huber { mu: 1.0 };
        loss.update_param(0.5);
        assert_eq!(loss.get_param(), 0.5);

        let mut loss = RobustLoss::Tukey { mu: 1.0 };
        loss.update_param(2.0);
        assert_eq!(loss.get_param(), 2.0);
    }

    #[test]
    fn test_compute_weight_same_as_weight() {
        let loss = RobustLoss::GemanMcClure { mu: 1.0 };
        for error in [0.1, 0.5, 1.0, 2.0, 5.0].iter() {
            assert_eq!(loss.weight(*error), loss.compute_weight(*error));
        }
    }
}
