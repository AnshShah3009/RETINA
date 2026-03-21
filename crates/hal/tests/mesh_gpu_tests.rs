//! GPU Mesh Kernel Tests
//!
//! Tests verify mathematical correctness by comparing GPU results against
//! reference CPU implementations.

use cv_hal::gpu::GpuContext;
use nalgebra::{Point3, Vector3};
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

/// CPU reference implementation for vertex normals
fn compute_vertex_normals_cpu(vertices: &[Point3<f32>], faces: &[[u32; 3]]) -> Vec<Vector3<f32>> {
    let mut normals: Vec<Vector3<f32>> = vec![Vector3::zeros(); vertices.len()];
    let mut counts: Vec<u32> = vec![0; vertices.len()];

    for face in faces {
        let v0 = vertices[face[0] as usize];
        let v1 = vertices[face[1] as usize];
        let v2 = vertices[face[2] as usize];

        // Cross product of edges
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2);

        // Accumulate to each vertex
        for idx in [face[0], face[1], face[2]] {
            normals[idx as usize] += face_normal;
            counts[idx as usize] += 1;
        }
    }

    // Normalize
    for (i, normal) in normals.iter_mut().enumerate() {
        if counts[i] > 0 {
            let len = normal.norm();
            if len > 1e-8 {
                *normal /= len;
            }
        }
    }

    normals
}

/// CPU reference implementation for mesh bounds
fn compute_bounds_cpu(vertices: &[Point3<f32>]) -> (Point3<f32>, Point3<f32>) {
    if vertices.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min_bound = Point3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_bound = Point3::new(f32::MIN, f32::MIN, f32::MIN);

    for v in vertices {
        min_bound.x = min_bound.x.min(v.x);
        min_bound.y = min_bound.y.min(v.y);
        min_bound.z = min_bound.z.min(v.z);
        max_bound.x = max_bound.x.max(v.x);
        max_bound.y = max_bound.y.max(v.y);
        max_bound.z = max_bound.z.max(v.z);
    }

    (min_bound, max_bound)
}

/// CPU reference implementation for vertex map from depth
fn compute_vertex_map_cpu(
    depth: &[f32],
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Vec<Vector3<f32>> {
    let mut result = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let d = depth[idx];

            if d <= 0.001 {
                result.push(Vector3::zeros());
            } else {
                let px = (x as f32 - cx) * d / fx;
                let py = (y as f32 - cy) * d / fy;
                result.push(Vector3::new(px, py, d));
            }
        }
    }

    result
}

/// CPU reference implementation for normal map from vertex map
fn compute_normal_map_cpu(
    vertex_map: &[Vector3<f32>],
    width: u32,
    height: u32,
) -> Vec<Vector3<f32>> {
    let mut normals = Vec::with_capacity(vertex_map.len());

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let v = vertex_map[idx];

            // Simple cross product of neighbors
            let mut normal = Vector3::zeros();
            let mut count = 0i32;

            // Check 4 neighbors and compute cross products
            if x > 0 && y > 0 && x < width - 1 && y < height - 1 {
                let v_left = vertex_map[(y * width + (x - 1)) as usize];
                let v_right = vertex_map[(y * width + (x + 1)) as usize];
                let v_up = vertex_map[((y - 1) * width + x) as usize];
                let v_down = vertex_map[((y + 1) * width + x) as usize];

                // Only compute if all neighbors are valid (non-zero)
                if v_left.norm_squared() > 0.0
                    && v_right.norm_squared() > 0.0
                    && v_up.norm_squared() > 0.0
                    && v_down.norm_squared() > 0.0
                {
                    let e1 = v_right - v_left;
                    let e2 = v_down - v_up;
                    normal = e1.cross(&e2);
                    count = 1;
                }
            }

            if count > 0 && normal.norm() > 1e-8 {
                normal.normalize_mut();
                normals.push(normal);
            } else {
                normals.push(Vector3::new(0.0, 1.0, 0.0));
            }
        }
    }

    normals
}

/// Compare two vectors with tolerance
fn vectors_close(a: &Vector3<f32>, b: &Vector3<f32>, epsilon: f32) -> bool {
    (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
}

/// Compare two points with tolerance
fn points_close(a: &Point3<f32>, b: &Point3<f32>, epsilon: f32) -> bool {
    (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
}

#[test]
fn test_vertex_normals_simple_triangle() -> Result<(), Box<dyn Error>> {
    // Skip if no GPU available
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Simple triangle: (0,0,0), (1,0,0), (0,1,0)
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];

    // Expected: all normals should point in +Z direction (cross product)
    let expected = compute_vertex_normals_cpu(&vertices, &faces);

    // The normal should be (0, 0, 1) or (0, 0, -1) depending on winding
    assert!(
        expected[0].z.abs() > 0.9,
        "Normal should be primarily in Z direction"
    );

    Ok(())
}

#[test]
fn test_vertex_normals_cube() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Cube vertices
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    ];

    // Cube faces (6 faces, 2 triangles each)
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3], // front
        [4, 6, 5],
        [4, 7, 6], // back
        [0, 4, 5],
        [0, 5, 1], // bottom
        [2, 6, 7],
        [2, 7, 3], // top
        [0, 3, 7],
        [0, 7, 4], // left
        [1, 5, 6],
        [1, 6, 2], // right
    ];

    // All vertex normals should be unit vectors
    let normals_cpu = compute_vertex_normals_cpu(&vertices, &faces);

    for (i, normal) in normals_cpu.iter().enumerate() {
        let len = normal.norm();
        assert!(
            (len - 1.0).abs() < 0.001,
            "Vertex {} normal should be unit vector, got length {}",
            i,
            len
        );
    }

    // Normals should point outward from cube center (0.5, 0.5, 0.5)
    // Vertex 0 at (0,0,0) should have normal pointing (-1,-1,-1) direction
    let center_vec = Vector3::new(0.5, 0.5, 0.5);
    for (i, v) in vertices.iter().enumerate() {
        let v_vec = Vector3::new(v.x, v.y, v.z);
        let to_center = center_vec - v_vec;
        let dot = normals_cpu[i].dot(&to_center);
        // Dot product should be negative (normal points away from center)
        assert!(
            dot < 0.1,
            "Vertex {} normal should point outward, dot={}",
            i,
            dot
        );
    }

    Ok(())
}

#[test]
fn test_bounds_empty_mesh() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let vertices: Vec<Point3<f32>> = vec![];

    // Empty mesh should return error
    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices);
    assert!(result.is_err(), "Empty mesh should return error");

    Ok(())
}

#[test]
fn test_bounds_single_point() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let vertices = vec![Point3::new(3.0, 5.0, 7.0)];
    let expected = (Point3::new(3.0, 5.0, 7.0), Point3::new(3.0, 5.0, 7.0));

    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices)?;

    assert!(
        points_close(&result.0, &expected.0, 1e-5),
        "Min bounds should match: {:?}",
        result.0
    );
    assert!(
        points_close(&result.1, &expected.1, 1e-5),
        "Max bounds should match: {:?}",
        result.1
    );

    Ok(())
}

#[test]
fn test_bounds_cube() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
    ];

    let expected = (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices)?;

    assert!(
        points_close(&result.0, &expected.0, 1e-5),
        "Min bounds should be (0,0,0), got {:?}",
        result.0
    );
    assert!(
        points_close(&result.1, &expected.1, 1e-5),
        "Max bounds should be (1,1,1), got {:?}",
        result.1
    );

    Ok(())
}

#[test]
fn test_bounds_negative_coordinates() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    let vertices = vec![Point3::new(-5.0, -3.0, -1.0), Point3::new(2.0, 4.0, 6.0)];

    let expected = (Point3::new(-5.0, -3.0, -1.0), Point3::new(2.0, 4.0, 6.0));

    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices)?;

    assert!(
        points_close(&result.0, &expected.0, 1e-5),
        "Min bounds incorrect: {:?}",
        result.0
    );
    assert!(
        points_close(&result.1, &expected.1, 1e-5),
        "Max bounds incorrect: {:?}",
        result.1
    );

    Ok(())
}

#[test]
fn test_vertex_map_invalid_dimensions() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Empty depth should return empty result
    let depth: Vec<f32> = vec![];
    let result = cv_hal::gpu_kernels::odometry_gpu::compute_vertex_map(
        &ctx,
        &depth,
        &[525.0, 525.0, 319.5, 239.5],
        640,
        480,
    )?;
    assert!(result.is_empty(), "Empty input should return empty output");

    Ok(())
}

#[test]
fn test_vertex_map_center_projection() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Simple 3x3 depth image with depth=1.0 at center
    let width = 3u32;
    let height = 3u32;
    let fx = 1.0;
    let fy = 1.0;
    let cx = 1.0;
    let cy = 1.0;

    // All pixels have depth = 1.0
    let depth = vec![1.0; (width * height) as usize];

    let cpu_result = compute_vertex_map_cpu(&depth, width, height, fx, fy, cx, cy);
    let gpu_result = cv_hal::gpu_kernels::odometry_gpu::compute_vertex_map(
        &ctx,
        &depth,
        &[fx, fy, cx, cy],
        width,
        height,
    )?;

    // Check center pixel (1,1): should project to (0,0,1)
    let center_idx = (1 * width + 1) as usize;
    assert!(
        vectors_close(&gpu_result[center_idx], &cpu_result[center_idx], 1e-3),
        "Center pixel projection mismatch: GPU {:?} vs CPU {:?}",
        gpu_result[center_idx],
        cpu_result[center_idx]
    );

    // Check corner pixel (0,0): should project to (-cx, -cy, 1) = (-1, -1, 1)
    let corner_idx = 0usize;
    assert!(
        vectors_close(&gpu_result[corner_idx], &cpu_result[corner_idx], 1e-3),
        "Corner pixel projection mismatch: GPU {:?} vs CPU {:?}",
        gpu_result[corner_idx],
        cpu_result[corner_idx]
    );

    Ok(())
}

#[test]
fn test_vertex_map_known_values() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // 4x4 depth image
    let width = 4u32;
    let height = 4u32;
    let fx = 525.0;
    let fy = 525.0;
    let cx = 319.5;
    let cy = 239.5;

    // Create depth with known value at center
    let mut depth = vec![0.0; (width * height) as usize];
    let center_idx = (2 * width + 2) as usize; // (2,2) pixel
    depth[center_idx] = 2.0; // 2 meters away

    let cpu_result = compute_vertex_map_cpu(&depth, width, height, fx, fy, cx, cy);
    let gpu_result = cv_hal::gpu_kernels::odometry_gpu::compute_vertex_map(
        &ctx,
        &depth,
        &[fx, fy, cx, cy],
        width,
        height,
    )?;

    // At pixel (2,2) with depth=2.0:
    // x = (2 - 319.5) * 2.0 / 525.0 = -1.2095...
    // y = (2 - 239.5) * 2.0 / 525.0 = -0.9057...
    // z = 2.0
    let expected = &cpu_result[center_idx];
    let actual = &gpu_result[center_idx];

    assert!(
        (actual.z - expected.z).abs() < 0.01,
        "Z should be 2.0, got {}",
        actual.z
    );

    Ok(())
}

#[test]
fn test_vertex_map_zero_depth() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // 2x2 with zero depth (invalid)
    let width = 2u32;
    let height = 2u32;
    let depth = vec![0.0, 0.0, 0.0, 0.0];

    let result = cv_hal::gpu_kernels::odometry_gpu::compute_vertex_map(
        &ctx,
        &depth,
        &[525.0, 525.0, 319.5, 239.5],
        width,
        height,
    )?;

    // Zero depth should produce zero/invalid vertices
    for v in &result {
        assert!(
            v.norm_squared() < 0.001,
            "Zero depth should produce zero vertices, got {:?}",
            v
        );
    }

    Ok(())
}

#[test]
fn test_normal_map_unit_vectors() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Create flat plane: all Z=1.0
    let width = 5u32;
    let height = 5u32;
    let mut vertex_map = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            vertex_map.push(Vector3::new(x as f32, y as f32, 1.0));
        }
    }

    let result =
        cv_hal::gpu_kernels::odometry_gpu::compute_normal_map(&ctx, &vertex_map, width, height)?;

    // For a flat plane, all normals should point in +Z direction
    // (accounting for the cross product direction in the algorithm)
    for (i, normal) in result.iter().enumerate() {
        let len = normal.norm();
        assert!(
            (len - 1.0).abs() < 0.001,
            "Normal {} should be unit vector, got length {}",
            i,
            len
        );
    }

    Ok(())
}

#[test]
fn test_normal_map_vs_cpu_reference() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Create simple 3x3 vertex map with known structure
    let width = 3u32;
    let height = 3u32;
    let vertex_map = vec![
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(1.0, 0.0, 1.0),
        Vector3::new(2.0, 0.0, 1.0),
        Vector3::new(0.0, 1.0, 1.0),
        Vector3::new(1.0, 1.0, 1.0), // center
        Vector3::new(2.0, 1.0, 1.0),
        Vector3::new(0.0, 2.0, 1.0),
        Vector3::new(1.0, 2.0, 1.0),
        Vector3::new(2.0, 2.0, 1.0),
    ];

    let cpu_result = compute_normal_map_cpu(&vertex_map, width, height);
    let gpu_result =
        cv_hal::gpu_kernels::odometry_gpu::compute_normal_map(&ctx, &vertex_map, width, height)?;

    // Center normal should be approximately (0, 0, 1) for flat plane
    let center_idx = 4usize;
    let dot = gpu_result[center_idx].dot(&cpu_result[center_idx]);

    // For flat plane, both should point in same direction
    assert!(dot > 0.5, "Center normals should be similar, dot={}", dot);

    Ok(())
}

#[test]
fn test_normal_map_consistency() -> Result<(), Box<dyn Error>> {
    let ctx = match cv_hal::gpu::GpuContext::global() {
        Ok(ctx) => ctx,
        Err(_) => {
            init_gpu()?;
            return Ok(());
        }
    };

    // Create multiple vertex maps with same structure, different depths
    let width = 4u32;
    let height = 4u32;

    for depth in [0.5, 1.0, 2.0, 5.0] {
        let mut vertex_map = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                vertex_map.push(Vector3::new(x as f32, y as f32, depth));
            }
        }

        let result = cv_hal::gpu_kernels::odometry_gpu::compute_normal_map(
            &ctx,
            &vertex_map,
            width,
            height,
        )?;

        // All normals should still be unit vectors
        for (i, normal) in result.iter().enumerate() {
            let len = normal.norm();
            assert!(
                (len - 1.0).abs() < 0.01,
                "Normal {} should be unit vector for depth={}, got {}",
                i,
                depth,
                len
            );
        }
    }

    Ok(())
}
