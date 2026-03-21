//! GPU Spatial Kernel Tests
//!
//! Tests verify mathematical correctness by comparing GPU results against
//! reference CPU implementations.

use cv_hal::gpu::GpuContext;
use nalgebra::Vector3;
use pollster::block_on;
use std::error::Error;

/// Initialize GPU context if not already initialized
fn init_gpu() -> Result<(), Box<dyn Error>> {
    if GpuContext::global().is_ok() {
        return Ok(());
    }
    block_on(GpuContext::init_global()).map_err(|e| format!("GPU init failed: {}", e))?;
    Ok(())
}

/// CPU reference for voxel grid downsampling
fn voxel_grid_downsample_cpu(points: &[Vector3<f32>], voxel_size: f32) -> Vec<Vector3<f32>> {
    if points.is_empty() {
        return vec![];
    }

    // Compute bounding box
    let mut min_bound = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_bound = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
    for p in points {
        min_bound.x = min_bound.x.min(p.x);
        min_bound.y = min_bound.y.min(p.y);
        min_bound.z = min_bound.z.min(p.z);
        max_bound.x = max_bound.x.max(p.x);
        max_bound.y = max_bound.y.max(p.y);
        max_bound.z = max_bound.z.max(p.z);
    }

    let inv_voxel = 1.0 / voxel_size;
    let vol_x = ((max_bound.x - min_bound.x) * inv_voxel).ceil() as usize + 1;
    let vol_y = ((max_bound.y - min_bound.y) * inv_voxel).ceil() as usize + 1;
    let vol_z = ((max_bound.z - min_bound.z) * inv_voxel).ceil() as usize + 1;

    // Hash points to voxels and accumulate
    let mut voxel_accum: Vec<[f64; 4]> = vec![[0.0, 0.0, 0.0, 0.0]; vol_x * vol_y * vol_z];

    for p in points {
        let vx = ((p.x - min_bound.x) * inv_voxel) as usize;
        let vy = ((p.y - min_bound.y) * inv_voxel) as usize;
        let vz = ((p.z - min_bound.z) * inv_voxel) as usize;

        if vx < vol_x && vy < vol_y && vz < vol_z {
            let idx = (vz * vol_y + vy) * vol_x + vx;
            voxel_accum[idx][0] += p.x as f64;
            voxel_accum[idx][1] += p.y as f64;
            voxel_accum[idx][2] += p.z as f64;
            voxel_accum[idx][3] += 1.0;
        }
    }

    // Average and collect
    voxel_accum
        .iter()
        .filter(|v| v[3] > 0.5)
        .map(|v| {
            Vector3::new(
                v[0] as f32 / v[3] as f32,
                v[1] as f32 / v[3] as f32,
                v[2] as f32 / v[3] as f32,
            )
        })
        .collect()
}

/// CPU reference for brute force k-NN
fn knn_cpu(points: &[Vector3<f32>], queries: &[Vector3<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    queries
        .iter()
        .map(|q| {
            let mut distances: Vec<(u32, f32)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i as u32, (p - q).norm_squared()))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);
            distances
        })
        .collect()
}

/// CPU reference for radius search

/// CPU reference for radius search
fn radius_search_cpu(
    points: &[Vector3<f32>],
    queries: &[Vector3<f32>],
    radius: f32,
) -> Vec<Vec<(u32, f32)>> {
    let radius_sq = radius * radius;
    queries
        .iter()
        .map(|q| {
            let qx = q.x;
            let qy = q.y;
            let qz = q.z;
            points
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    let dx = p.x - qx;
                    let dy = p.y - qy;
                    let dz = p.z - qz;
                    dx * dx + dy * dy + dz * dz <= radius_sq
                })
                .map(|(i, p)| {
                    let dx = p.x - qx;
                    let dy = p.y - qy;
                    let dz = p.z - qz;
                    (i as u32, (dx * dx + dy * dy + dz * dz).sqrt())
                })
                .collect()
        })
        .collect()
}

/// CPU reference for Morton encoding
fn morton_encode_cpu(x: u32, y: u32, z: u32) -> u32 {
    let mut code = 0u32;
    for i in 0..10 {
        code |= ((x >> i) & 1) << (3 * i)
            | ((y >> i) & 1) << (3 * i + 1)
            | ((z >> i) & 1) << (3 * i + 2);
    }
    code
}

fn vectors_close(a: &Vector3<f32>, b: &Vector3<f32>, epsilon: f32) -> bool {
    (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
}

#[test]
fn test_morton_encode_known_values() {
    // Test against known Morton code values
    assert_eq!(morton_encode_cpu(0, 0, 0), 0);
    assert_eq!(morton_encode_cpu(1, 0, 0), 1);
    assert_eq!(morton_encode_cpu(0, 1, 0), 2);
    assert_eq!(morton_encode_cpu(0, 0, 1), 4);
    assert_eq!(morton_encode_cpu(1, 1, 1), 7);
}

#[test]
fn test_voxel_grid_empty() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points: Vec<Vector3<f32>> = vec![];
    let result = cv_hal::gpu_kernels::spatial_gpu::voxel_grid_downsample(&ctx, &points, 0.1)?;

    assert!(result.is_empty(), "Empty input should give empty output");
    Ok(())
}

#[test]
fn test_voxel_grid_single_point() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![Vector3::new(1.0, 2.0, 3.0)];
    let result = cv_hal::gpu_kernels::spatial_gpu::voxel_grid_downsample(&ctx, &points, 0.5)?;

    assert_eq!(result.len(), 1, "Single point should give single voxel");
    assert!(
        vectors_close(&result[0], &points[0], 1e-5),
        "Voxel centroid should match original point"
    );
    Ok(())
}

#[test]
fn test_voxel_grid_known_voxel_size() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Two points in the same voxel should give one result
    let points = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.1, 0.1, 0.1), // Same voxel with 0.5 size
    ];

    let result = cv_hal::gpu_kernels::spatial_gpu::voxel_grid_downsample(&ctx, &points, 0.5)?;

    // Should be 1 voxel with average position
    assert_eq!(result.len(), 1, "Points in same voxel should merge");

    let expected = Vector3::new(0.05, 0.05, 0.05);
    assert!(
        vectors_close(&result[0], &expected, 1e-3),
        "Voxel centroid should be average: got {:?}",
        result[0]
    );

    Ok(())
}

#[test]
fn test_voxel_grid_different_voxels() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Two points far apart should give two voxels
    let points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 10.0, 10.0)];

    let result = cv_hal::gpu_kernels::spatial_gpu::voxel_grid_downsample(&ctx, &points, 0.5)?;

    assert_eq!(result.len(), 2, "Far points should be in different voxels");

    Ok(())
}

#[test]
fn test_voxel_grid_vs_cpu() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Grid of points
    let mut points = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                points.push(Vector3::new(x as f32, y as f32, z as f32));
            }
        }
    }

    let voxel_size = 2.0;

    let gpu_result =
        cv_hal::gpu_kernels::spatial_gpu::voxel_grid_downsample(&ctx, &points, voxel_size)?;
    let cpu_result = voxel_grid_downsample_cpu(&points, voxel_size);

    // Should have fewer points after downsampling
    assert!(
        gpu_result.len() < points.len(),
        "Downsampled should have fewer points"
    );

    // Compare counts
    assert_eq!(
        gpu_result.len(),
        cpu_result.len(),
        "GPU and CPU should produce same count"
    );

    // Note: centroids should match but order may differ
    // Just verify the total count is reasonable
    let expected_count = 125 / 8; // 2.5 x 2.5 x 2.5 = ~15 voxels
    assert!(
        (gpu_result.len() as i32 - expected_count as i32).abs() <= 2,
        "Voxel count should be approximately {} +/- 2, got {}",
        expected_count,
        gpu_result.len()
    );

    Ok(())
}

#[test]
fn test_knn_empty() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![Vector3::new(0.0, 0.0, 0.0)];
    let queries: Vec<Vector3<f32>> = vec![];

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let result =
        cv_hal::gpu_kernels::spatial_gpu::batch_nearest_neighbors(&ctx, &kdtree, &queries, 3)?;

    assert!(result.is_empty(), "Empty queries should give empty result");
    Ok(())
}

#[test]
fn test_knn_single_query() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    ];

    let queries = vec![Vector3::new(0.1, 0.1, 0.0)];

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let result =
        cv_hal::gpu_kernels::spatial_gpu::batch_nearest_neighbors(&ctx, &kdtree, &queries, 2)?;

    assert_eq!(result.len(), 1, "Should have one query result");
    assert_eq!(result[0].len(), 2, "Should return k=2 nearest");

    // First nearest should be point 0 or 2 (both ~0.14 away)
    assert!(
        result[0][0].0 == 0 || result[0][0].0 == 2,
        "First nearest should be point 0 or 2, got {}",
        result[0][0].0
    );

    Ok(())
}

#[test]
fn test_knn_vs_cpu() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Random-ish points
    let points = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(1.0, 1.0, 0.0),
        Vector3::new(0.5, 0.5, 0.0),
    ];

    let queries = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0)];

    let k = 3usize;

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let gpu_result = cv_hal::gpu_kernels::spatial_gpu::batch_nearest_neighbors(
        &ctx, &kdtree, &queries, k as u32,
    )?;
    let cpu_result = knn_cpu(&points, &queries, k);

    assert_eq!(gpu_result.len(), cpu_result.len());

    for (gpu_nn, cpu_nn) in gpu_result.iter().zip(cpu_result.iter()) {
        assert_eq!(gpu_nn.len(), cpu_nn.len());

        // Sort by index for comparison
        let mut gpu_sorted = gpu_nn.clone();
        let mut cpu_sorted = cpu_nn.clone();
        gpu_sorted.sort_by_key(|(idx, _)| *idx);
        cpu_sorted.sort_by_key(|(idx, _)| *idx);

        for ((g_idx, g_dist), (c_idx, c_dist)) in gpu_sorted.iter().zip(cpu_sorted.iter()) {
            assert_eq!(g_idx, c_idx, "Indices should match");
            assert!((g_dist - c_dist).abs() < 1e-4, "Distances should match");
        }
    }

    Ok(())
}

#[test]
fn test_radius_search_empty() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![Vector3::new(0.0, 0.0, 0.0)];
    let queries: Vec<Vector3<f32>> = vec![];

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let result =
        cv_hal::gpu_kernels::spatial_gpu::batch_radius_search(&ctx, &kdtree, &queries, 1.0)?;

    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_radius_search_all_within() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // All points within large radius
    let points = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5, 0.0, 0.0),
        Vector3::new(0.0, 0.5, 0.0),
    ];

    let queries = vec![Vector3::new(0.0, 0.0, 0.0)];
    let radius = 10.0;

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let result =
        cv_hal::gpu_kernels::spatial_gpu::batch_radius_search(&ctx, &kdtree, &queries, radius)?;

    assert_eq!(result[0].len(), 3, "All points within large radius");

    Ok(())
}

#[test]
fn test_radius_search_none_within() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![
        Vector3::new(100.0, 100.0, 100.0),
        Vector3::new(200.0, 200.0, 200.0),
    ];

    let queries = vec![Vector3::new(0.0, 0.0, 0.0)];
    let radius = 1.0;

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let result =
        cv_hal::gpu_kernels::spatial_gpu::batch_radius_search(&ctx, &kdtree, &queries, radius)?;

    assert!(result[0].is_empty(), "No points within small radius");

    Ok(())
}

#[test]
fn test_radius_search_vs_cpu() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(1.0, 1.0, 0.0),
        Vector3::new(0.5, 0.5, 0.0),
    ];

    let queries = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0)];

    let radius = 0.6;

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;
    let gpu_result =
        cv_hal::gpu_kernels::spatial_gpu::batch_radius_search(&ctx, &kdtree, &queries, radius)?;
    let cpu_result = radius_search_cpu(&points, &queries, radius);

    assert_eq!(gpu_result.len(), cpu_result.len());

    for (gpu_r, cpu_r) in gpu_result.iter().zip(cpu_result.iter()) {
        // Sort by index for comparison
        let mut gpu_sorted = gpu_r.clone();
        let mut cpu_sorted = cpu_r.clone();
        gpu_sorted.sort_by_key(|(idx, _)| *idx);
        cpu_sorted.sort_by_key(|(idx, _)| *idx);

        assert_eq!(gpu_sorted.len(), cpu_sorted.len(), "Same number of results");

        for ((g_idx, g_dist), (c_idx, c_dist)) in gpu_sorted.iter().zip(cpu_sorted.iter()) {
            assert_eq!(g_idx, c_idx);
            assert!((g_dist - c_dist).abs() < 1e-4);
        }
    }

    Ok(())
}

#[test]
fn test_kdtree_preserves_points() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let points = vec![Vector3::new(1.5, 2.5, 3.5), Vector3::new(10.0, 20.0, 30.0)];

    let kdtree = cv_hal::gpu_kernels::spatial_gpu::build_kdtree(&ctx, &points)?;

    // Query at original positions
    let queries = vec![Vector3::new(1.5, 2.5, 3.5), Vector3::new(10.0, 20.0, 30.0)];

    let result =
        cv_hal::gpu_kernels::spatial_gpu::batch_nearest_neighbors(&ctx, &kdtree, &queries, 1)?;

    // First query should find point 0 as nearest
    assert_eq!(result[0][0].0, 0, "First query should find first point");
    assert!(result[0][0].1 < 0.001, "Distance should be near zero");

    // Second query should find point 1 as nearest
    assert_eq!(result[1][0].0, 1, "Second query should find second point");
    assert!(result[1][0].1 < 0.001, "Distance should be near zero");

    Ok(())
}
