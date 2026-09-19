//! Example: Kalman Filter
//!
//! Run with: cargo run --example kalman_filter

fn main() {
    println!("=== cv-slam Kalman Filter Example ===\n");

    println!("1. Kalman Filter for State Estimation:");
    println!("   - 1D/2D position tracking");
    println!("   - Velocity estimation");
    println!("   - Sensor fusion support");

    println!("\n2. Usage:");
    println!("   use cv_slam::KalmanFilter;");
    println!("   let mut kf = KalmanFilter::<4>::new();");
    println!("   kf.predict(...);");
    println!("   kf.update(...);");

    println!("\nNote: Full Kalman filter API available in cv-slam crate");
    println!("      See scientific crate for optimization algorithms.");

    println!("\n=== Example Complete ===");
}
