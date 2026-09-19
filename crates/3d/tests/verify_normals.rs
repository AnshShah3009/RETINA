use cv_3d::normals::{estimate_normals_approx_integral, estimate_normals_cpu};
use nalgebra::{Point3, Vector3};
use std::time::Instant;

#[test]
fn verify_approx_normals_accuracy() {
    println!("--- Verifying Approx Normals Accuracy ---");

    // Create a sphere of points
    let radius = 10.0;
    let mut points = Vec::new();
    let mut ground_truth_normals = Vec::new();

    let stacks = 100;
    let sectors = 100;

    for i in 0..=stacks {
        let phi = std::f32::consts::PI * (i as f32 / stacks as f32);
        for j in 0..sectors {
            let theta = 2.0 * std::f32::consts::PI * (j as f32 / sectors as f32);

            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.sin() * theta.sin();
            let z = radius * phi.cos();

            let p = Point3::new(x, y, z);
            points.push(p);

            // Ground truth normal for a sphere at origin is just the normalized position
            ground_truth_normals.push(Vector3::new(x, y, z).normalize());
        }
    }

    println!("Testing with {} points on a sphere.", points.len());

    // 1. PCA Normals (Baseline)
    let start = Instant::now();
    let normals_pca = estimate_normals_cpu(&points, 15);
    println!("PCA Normals Time: {:?}", start.elapsed());

    // 2. Approx Integral Normals
    let start = Instant::now();
    let normals_integral = estimate_normals_approx_integral(&points);
    println!("Approx Integral Normals Time: {:?}", start.elapsed());

    // Compare
    let mut total_error_pca = 0.0;
    let mut total_error_integral = 0.0;
    let mut flipped_integral = 0;
    let mut degenerate_integral = 0;

    for i in 0..points.len() {
        let gt = ground_truth_normals[i];
        let n_pca = normals_pca[i];
        let n_integral = normals_integral[i];

        // PCA error (handle potential flips as PCA is sign-ambiguous)
        let dot_pca = gt.dot(&n_pca).abs();
        total_error_pca += dot_pca.acos();

        // Integral error
        if n_integral == Vector3::z() {
            degenerate_integral += 1;
        } else {
            let dot_integral = gt.dot(&n_integral);
            if dot_integral < 0.0 {
                flipped_integral += 1;
            }
            total_error_integral += dot_integral.abs().acos();
        }
    }

    let n = points.len() as f32;
    println!(
        "PCA Average Angle Error:      {:.4} rad",
        total_error_pca / n
    );
    println!(
        "Approx Integral Avg Error:    {:.4} rad (Degenerate: {})",
        total_error_integral / (n - degenerate_integral as f32),
        degenerate_integral
    );
    println!(
        "Approx Integral Flipped:      {} / {}",
        flipped_integral,
        points.len()
    );

    // Assertions to catch major regressions or bugs
    assert!(total_error_pca / n < 0.1);
    assert!(total_error_integral / n < 0.2);
}

#[test]
fn test_gpu_vs_cpu_normals() {
    eprintln!("\n=== GPU vs CPU Normals Test ===\n");

    // Create test points (sphere surface)
    let n = 1000;
    let points: Vec<Point3<f32>> = (0..n)
        .map(|i| {
            let theta = (i as f32) * 0.01;
            let phi = (i as f32) * 0.005;
            let r = 1.0;
            Point3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            )
        })
        .collect();

    eprintln!("Testing with {} points", points.len());

    // Test GPU path first (to see logs)
    eprintln!("--- Testing GPU/Auto Path ---");
    let gpu_normals = cv_3d::normals::estimate_normals_auto(&points, 15);
    eprintln!("GPU/Auto normals computed: {} normals", gpu_normals.len());

    // Test CPU path
    eprintln!("--- Testing CPU Path ---");
    let cpu_normals = cv_3d::normals::estimate_normals_cpu(&points, 15);
    eprintln!("CPU normals computed: {} normals", cpu_normals.len());

    // Compare (they should be similar, allowing for sign flips)
    let mut total_diff = 0.0f32;
    for (cn, gn) in cpu_normals.iter().zip(gpu_normals.iter()) {
        let diff = (cn - gn).norm();
        let diff_flipped = (cn + gn).norm(); // Allow sign flip
        total_diff += diff.min(diff_flipped);
    }

    let avg_diff = total_diff / n as f32;
    eprintln!("Average difference: {:.6}", avg_diff);

    // Normals should be very close (within 0.1 radians)
    assert!(
        avg_diff < 0.1,
        "GPU normals differ too much from CPU: {}",
        avg_diff
    );
}

#[test]
fn test_hybrid_normals() {
    eprintln!("\n=== Hybrid Normals Test (CPU kNN + GPU PCA) ===\n");

    let n = 1000;
    let points: Vec<Point3<f32>> = (0..n)
        .map(|i| {
            let theta = (i as f32) * 0.01;
            let phi = (i as f32) * 0.005;
            Point3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect();

    eprintln!("Testing with {} points", points.len());

    let hybrid_normals = cv_3d::normals::estimate_normals_hybrid(&points, 15);
    eprintln!("Hybrid normals computed: {} normals", hybrid_normals.len());

    let cpu_normals = cv_3d::normals::estimate_normals_cpu(&points, 15);
    eprintln!("CPU normals computed: {} normals", cpu_normals.len());

    // Compare
    let mut total_diff = 0.0f32;
    for (cn, hn) in cpu_normals.iter().zip(hybrid_normals.iter()) {
        let diff = (cn - hn).norm();
        let diff_flipped = (cn + hn).norm();
        total_diff += diff.min(diff_flipped);
    }

    let avg_diff = total_diff / n as f32;
    eprintln!("Average difference: {:.6}", avg_diff);

    assert!(
        avg_diff < 0.1,
        "Hybrid normals differ too much from CPU: {}",
        avg_diff
    );
}

#[test]
fn test_all_normals_paths() {
    eprintln!("\n=== Testing All Normals Paths ===\n");

    let n = 1000;
    let points: Vec<Point3<f32>> = (0..n)
        .map(|i| {
            let theta = (i as f32) * 0.01;
            let phi = (i as f32) * 0.005;
            Point3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect();

    eprintln!("Testing with {} points", points.len());

    // CPU voxel hash
    let t0 = std::time::Instant::now();
    let normals_voxel = cv_3d::normals::estimate_normals_cpu(&points, 15);
    eprintln!(
        "[CPU-VOXEL] {} normals in {:?}",
        normals_voxel.len(),
        t0.elapsed()
    );

    // CPU KDTree
    let t1 = std::time::Instant::now();
    let normals_kdtree = cv_3d::normals::estimate_normals_cpu_kdtree(&points, 15);
    eprintln!(
        "[CPU-KDTREE] {} normals in {:?}",
        normals_kdtree.len(),
        t1.elapsed()
    );

    // Hybrid
    let t2 = std::time::Instant::now();
    let normals_hybrid = cv_3d::normals::estimate_normals_hybrid(&points, 15);
    eprintln!(
        "[HYBRID] {} normals in {:?}",
        normals_hybrid.len(),
        t2.elapsed()
    );

    // GPU/Auto
    let t3 = std::time::Instant::now();
    let normals_auto = cv_3d::normals::estimate_normals_auto(&points, 15);
    eprintln!(
        "[GPU-AUTO] {} normals in {:?}",
        normals_auto.len(),
        t3.elapsed()
    );

    // Compare all to voxel hash
    let compare = |name: &str, normals: &[Vector3<f32>], ref_normals: &[Vector3<f32>]| {
        let mut total_diff = 0.0f32;
        for (r, n) in ref_normals.iter().zip(normals.iter()) {
            let diff = (*r - *n).norm();
            let diff_flipped = (*r + *n).norm();
            total_diff += diff.min(diff_flipped);
        }
        let avg_diff = total_diff / n as f32;
        eprintln!("[{}] Average difference vs voxel: {:.6}", name, avg_diff);
        avg_diff
    };

    let d1 = compare("KDTREE", &normals_kdtree, &normals_voxel);
    let d2 = compare("HYBRID", &normals_hybrid, &normals_voxel);
    let d3 = compare("AUTO", &normals_auto, &normals_voxel);

    // All paths should be very close
    assert!(d1 < 0.1, "KDTREE differs too much: {}", d1);
    assert!(d2 < 0.1, "HYBRID differs too much: {}", d2);
    assert!(d3 < 0.1, "AUTO differs too much: {}", d3);
}

#[test]
fn test_tiled_gpu_normals() {
    eprintln!("\n=== Tiled GPU Normals Test ===\n");

    let n = 1000;
    let points: Vec<Point3<f32>> = (0..n)
        .map(|i| {
            let theta = (i as f32) * 0.01;
            let phi = (i as f32) * 0.005;
            Point3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            )
        })
        .collect();

    eprintln!("Testing with {} points", points.len());

    // CPU voxel hash (reference)
    let normals_voxel = cv_3d::normals::estimate_normals_cpu(&points, 15);
    eprintln!("[CPU-VOXEL] {} normals", normals_voxel.len());

    // Tiled GPU
    let t0 = std::time::Instant::now();
    let normals_tiled = cv_3d::normals::estimate_normals_gpu_tiled(&points, 15);
    eprintln!(
        "[GPU-TILED] {} normals in {:?}",
        normals_tiled.len(),
        t0.elapsed()
    );

    // Compare
    let mut total_diff = 0.0f32;
    for (r, n) in normals_voxel.iter().zip(normals_tiled.iter()) {
        let diff = (*r - *n).norm();
        let diff_flipped = (*r + *n).norm();
        total_diff += diff.min(diff_flipped);
    }

    let avg_diff = total_diff / n as f32;
    eprintln!("[GPU-TILED] Average difference vs voxel: {:.6}", avg_diff);

    // Should be very close
    assert!(avg_diff < 0.1, "Tiled GPU differs too much: {}", avg_diff);
}
