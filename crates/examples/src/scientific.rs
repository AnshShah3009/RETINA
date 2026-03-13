//! Example: Scientific Computing (Integration & Special Functions)
//!
//! Run with: cargo run --example scientific

use cv_scientific::{integrate, mean, special, std, Interp1d};

fn main() {
    println!("=== Scientific Computing Example ===\n");

    // ===== Integration =====
    println!("--- Numerical Integration ---");

    // Integrate x^2 from 0 to 1 (should be 1/3 ≈ 0.333)
    let result = integrate::quad(|x| x * x, 0.0, 1.0);
    println!("∫x² dx from 0 to 1 = {:.6}", result.0);
    println!(
        "  Expected: 0.333333, Error: {:.6}",
        (result.0 - 1.0 / 3.0).abs()
    );

    // Integrate sin(x) from 0 to π (should be 2)
    let result = integrate::quad(|x| x.sin(), 0.0, std::f64::consts::PI);
    println!("∫sin(x) dx from 0 to π = {:.6}", result.0);
    println!("  Expected: 2.0, Error: {:.6}", (result.0 - 2.0).abs());

    // Trapezoid rule
    let result = integrate::trapezoid(|x| x * x, 0.0, 1.0, 100);
    println!("Trapezoid ∫x² dx from 0 to 1 = {:.6}", result);

    // ===== Special Functions =====
    println!("\n--- Special Functions ---");

    // Error function
    println!("erf(0.0) = {:.6}", special::erf(0.0));
    println!("erf(1.0) = {:.6}", special::erf(1.0));

    // Gamma function
    println!("Γ(1) = {:.0}", special::gamma(1.0));
    println!("Γ(5) = {:.0} (4! = 24)", special::gamma(5.0));
    println!("Γ(0.5) = {:.6} (√π ≈ 1.772)", special::gamma(0.5));

    // Bessel functions
    println!("J₀(0) = {:.6}", special::bessel_j0(0.0));
    println!("J₀(1) = {:.6}", special::bessel_j0(1.0));

    // ===== Statistics =====
    println!("\n--- Statistics ---");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("Data: {:?}", data);
    println!("Mean: {}", mean(&data).unwrap());
    println!("Std:  {}", std(&data).unwrap());

    // ===== Interpolation =====
    println!("\n--- Interpolation ---");

    let interp = Interp1d::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 1.0, 4.0, 9.0], // y = x²
    )
    .unwrap();

    println!("Interpolating x²:");
    println!("  x=0.0 → {:.3}", interp.call(0.0));
    println!("  x=0.5 → {:.3}", interp.call(0.5)); // Should be ~0.25
    println!("  x=1.5 → {:.3}", interp.call(1.5)); // Should be ~2.25
    println!("  x=2.5 → {:.3}", interp.call(2.5)); // Should be ~6.25

    println!("\n=== Example Complete ===");
}
