use cv_optimize::general::{minimize_nelder_mead, NelderMeadConfig};
use std::time::Instant;

#[test]
fn profile_optimize_algorithms() {
    println!("--- Profiling cv-optimize ---");

    // Define a simple n-dimensional Rosenbrock function for optimization
    let n = 100;
    let rosenbrock = |x: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..(x.len() - 1) {
            let t1 = 100.0 * (x[i + 1] - x[i] * x[i]).powi(2);
            let t2 = (1.0 - x[i]).powi(2);
            sum += t1 + t2;
        }
        sum
    };

    let initial_guess = vec![0.0; n];
    let config = NelderMeadConfig::default();

    let start = Instant::now();
    let result = minimize_nelder_mead(rosenbrock, &initial_guess, &config);
    let elapsed = start.elapsed();

    println!("Nelder-Mead (100D Rosenbrock): {:?}", elapsed);
    assert!(result.iterations > 0);
}
