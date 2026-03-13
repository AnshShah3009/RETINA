//! Example: Ray Casting
//!
//! Run with: cargo run --example raycasting

use cv_3d::{
    mesh::TriangleMesh,
    raycasting::{cast_ray_mesh, Ray, RayHit},
};
use nalgebra::{Point3, Vector3};

fn main() {
    println!("=== Ray Casting Example ===\n");

    // Create a simple triangle mesh (a flat triangle)
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];

    let mesh = TriangleMesh::with_vertices_and_faces(vertices, faces);

    println!("Created triangle mesh:");
    println!("  Vertices: {}", mesh.num_vertices());
    println!("  Faces: {}", mesh.num_faces());

    // Test ray that hits the triangle
    let ray1 = Ray::new(
        Point3::new(0.3, 0.3, 1.0),   // Origin above the triangle
        Vector3::new(0.0, 0.0, -1.0), // Pointing down
    );

    println!("\nRay 1: from (0.3, 0.3, 1.0) toward (0, 0, -1)");

    match cast_ray_mesh(&ray1, &mesh) {
        Some(hit) => {
            println!("  HIT!");
            println!("    Distance: {}", hit.distance);
            println!(
                "    Point: ({}, {}, {})",
                hit.point.x, hit.point.y, hit.point.z
            );
            println!(
                "    Normal: ({}, {}, {})",
                hit.normal.x, hit.normal.y, hit.normal.z
            );
            println!("    Triangle index: {}", hit.triangle_index);
        }
        None => println!("  No hit"),
    }

    // Test ray that misses the triangle
    let ray2 = Ray::new(
        Point3::new(0.7, 0.7, 1.0), // Origin above but outside
        Vector3::new(0.0, 0.0, -1.0),
    );

    println!("\nRay 2: from (0.7, 0.7, 1.0) toward (0, 0, -1)");

    match cast_ray_mesh(&ray2, &mesh) {
        Some(hit) => println!("  HIT at distance: {}", hit.distance),
        None => println!("  No hit (missed the triangle)"),
    }

    // Test ray parallel to the triangle plane
    let ray3 = Ray::new(
        Point3::new(0.3, 0.3, 1.0),
        Vector3::new(1.0, 0.0, 0.0), // Parallel to plane
    );

    println!("\nRay 3: from (0.3, 0.3, 1.0) toward (1, 0, 0) (parallel)");

    match cast_ray_mesh(&ray3, &mesh) {
        Some(hit) => println!("  HIT at distance: {}", hit.distance),
        None => println!("  No hit (ray parallel to triangle)"),
    }

    println!("\n=== Example Complete ===");
}
