//! Example: Bundle Adjustment
//!
//! Run with: cargo run --example bundle_adjustment

fn main() {
    println!("=== cv-sfm Bundle Adjustment Example ===\n");

    println!("1. Bundle Adjustment:");
    println!("   - Refines camera poses and 3D points");
    println!("   - Minimizes reprojection error");
    println!("   - Supports various optimization methods");

    println!("\n2. Structure from Motion Pipeline:");
    println!("   - Feature extraction");
    println!("   - Feature matching");
    println!("   - Triangulation");
    println!("   - Bundle adjustment");

    println!("\n3. Usage:");
    println!("   use cv_sfm::bundle_adjustment;");
    println!("   let result = bundle_adjustment(&problem);");

    println!("\nNote: Full API in cv-sfm and cv-optimize crates");
    println!("      See cv-optimize for Levenberg-Marquardt implementation.");

    println!("\n=== Example Complete ===");
}
