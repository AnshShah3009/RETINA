//! Example: Point Cloud Registration
//!
//! Run with: cargo run --example registration_icp

use cv_core::PointCloud;
use cv_registration::{registration_icp_point_to_plane, ICPResult};
use nalgebra::{Matrix4, Point3, Vector3};

fn main() {
    println!("=== ICP Registration Example ===\n");

    // Create source point cloud (a simple cube)
    let source_points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    ];

    // Create target point cloud (source transformed)
    let target_points = vec![
        Point3::new(0.1, 0.1, 0.0),
        Point3::new(1.1, 0.1, 0.0),
        Point3::new(1.1, 1.1, 0.0),
        Point3::new(0.1, 1.1, 0.0),
        Point3::new(0.1, 0.1, 1.0),
        Point3::new(1.1, 0.1, 1.0),
        Point3::new(1.1, 1.1, 1.0),
        Point3::new(0.1, 1.1, 1.0),
    ];

    // Add normals to target for point-to-plane ICP
    let target_normals = vec![
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];

    let source = PointCloud::new(source_points);
    let target = PointCloud::new(target_points)
        .with_normals(target_normals)
        .unwrap();

    println!("Source points: {}", source.points.len());
    println!("Target points: {}", target.points.len());

    // Run ICP registration
    let result: Option<ICPResult> = registration_icp_point_to_plane(
        &source,
        &target,
        1.0,                  // max correspondence distance
        &Matrix4::identity(), // initial transformation
        50,                   // max iterations
    );

    match result {
        Some(icp) => {
            println!("\nRegistration Results:");
            println!("  Fitness: {:.4}", icp.fitness);
            println!("  RMSE: {:.6}", icp.inlier_rmse);
            println!("  Iterations: {}", icp.num_iterations);
            println!("\nTransformation matrix:");
            for i in 0..4 {
                println!(
                    "  [{:.4}, {:.4}, {:.4}, {:.4}]",
                    icp.transformation[(i, 0)],
                    icp.transformation[(i, 1)],
                    icp.transformation[(i, 2)],
                    icp.transformation[(i, 3)]
                );
            }
        }
        None => {
            println!("Registration failed!");
        }
    }

    println!("\n=== Example Complete ===");
}
