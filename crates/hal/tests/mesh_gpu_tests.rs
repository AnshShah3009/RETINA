//! GPU Mesh Kernel Tests
//!
//! Tests verify mathematical correctness by comparing GPU results against
//! reference CPU implementations.

use cv_hal::gpu::GpuContext;
use nalgebra::{Point3, Vector3};
use pollster::block_on;
use std::error::Error;

/// Initialize GPU context if available. Returns None when no GPU (skip tests).
fn require_gpu() -> Result<Option<&'static GpuContext>, Box<dyn Error>> {
    if let Ok(ctx) = GpuContext::global() {
        return Ok(Some(ctx));
    }
    match block_on(GpuContext::init_global()) {
        Ok(ctx) => Ok(Some(ctx)),
        Err(e) => {
            println!("Skipping GPU test (no adapter): {}", e);
            Ok(None)
        }
    }
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
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let gpu = cv_hal::gpu_kernels::mesh_gpu::compute_vertex_normals(&ctx, &vertices, &faces)?;

    // The normal should be (0, 0, 1) or (0, 0, -1) depending on winding
    assert!(
        expected[0].z.abs() > 0.9,
        "Normal should be primarily in Z direction"
    );
    assert!(
        vectors_close(&expected[0], &gpu[0], 1e-3) || vectors_close(&expected[0], &(-gpu[0]), 1e-3),
        "GPU {:?} should match CPU {:?}",
        gpu[0],
        expected[0]
    );

    Ok(())
}

#[test]
fn test_vertex_normals_cube() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let normals_gpu =
        cv_hal::gpu_kernels::mesh_gpu::compute_vertex_normals(&ctx, &vertices, &faces)?;

    for (i, normal) in normals_cpu.iter().enumerate() {
        let len = normal.norm();
        assert!(
            (len - 1.0).abs() < 0.001,
            "Vertex {} normal should be unit vector, got length {}",
            i,
            len
        );
        assert!(
            vectors_close(normal, &normals_gpu[i], 1e-3)
                || vectors_close(normal, &(-normals_gpu[i]), 1e-3),
            "GPU normal {} {:?} should match CPU {:?} (up to sign)",
            i,
            normals_gpu[i],
            normal
        );
    }

    Ok(())
}

#[test]
fn test_bounds_empty_mesh() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
    };

    let vertices: Vec<Point3<f32>> = vec![];

    // Empty mesh should return error
    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices);
    assert!(result.is_err(), "Empty mesh should return error");

    Ok(())
}

#[test]
fn test_bounds_single_point() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
    };

    let vertices = vec![Point3::new(3.0, 5.0, 7.0)];

    // CONTROL: the trivial case, checked against a hand-written literal rather
    // than the reference, so a bug in the reference below cannot hide here.
    assert_eq!(compute_bounds_cpu(&vertices), (vertices[0], vertices[0]));

    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices)?;

    assert!(
        points_close(&result.0, &vertices[0], 1e-5),
        "Min bounds should match: {:?}",
        result.0
    );
    assert!(
        points_close(&result.1, &vertices[0], 1e-5),
        "Max bounds should match: {:?}",
        result.1
    );

    Ok(())
}

#[test]
fn test_bounds_cube() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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

    // CONTROL: the reference's own answer on this input is a literal, so the
    // comparison below is against a known value and not just self-consistency.
    assert_eq!(
        compute_bounds_cpu(&vertices),
        (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0))
    );

    // PARITY: the GPU reduction must agree with the CPU reduction elementwise.
    // A literal expectation can only be wrong about one configuration; the
    // reference covers whatever the vertex set actually is, so a shader that
    // drops or mis-indexes a vertex is caught on any input.
    let expected = compute_bounds_cpu(&vertices);

    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices)?;

    assert!(
        points_close(&result.0, &expected.0, 1e-5),
        "GPU min {:?} disagrees with CPU reference {:?}",
        result.0,
        expected.0
    );
    assert!(
        points_close(&result.1, &expected.1, 1e-5),
        "GPU max {:?} disagrees with CPU reference {:?}",
        result.1,
        expected.1
    );

    Ok(())
}

#[test]
fn test_bounds_negative_coordinates() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
    };

    let vertices = vec![Point3::new(-5.0, -3.0, -1.0), Point3::new(2.0, 4.0, 6.0)];

    // The negative-coordinate case is where a shader that initialises its
    // accumulator to +INF rather than the first vertex, or that uses an
    // unsigned comparison, diverges from the reference.
    let expected = compute_bounds_cpu(&vertices);
    assert_eq!(
        expected,
        (Point3::new(-5.0, -3.0, -1.0), Point3::new(2.0, 4.0, 6.0)),
        "the CPU reference itself must be right here, or the parity check below \
         is vacuous"
    );

    let result = cv_hal::gpu_kernels::mesh_gpu::compute_bounds(&ctx, &vertices)?;

    assert!(
        points_close(&result.0, &expected.0, 1e-5),
        "Min bounds incorrect: {:?} (reference {:?})",
        result.0,
        expected.0
    );
    assert!(
        points_close(&result.1, &expected.1, 1e-5),
        "Max bounds incorrect: {:?} (reference {:?})",
        result.1,
        expected.1
    );

    Ok(())
}

#[test]
fn test_vertex_map_invalid_dimensions() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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

    // Center normal should be approximately (0, 0, ±1) for flat plane (winding may flip sign)
    let center_idx = 4usize;
    let dot = gpu_result[center_idx].dot(&cpu_result[center_idx]);

    // For flat plane, normals should align up to sign
    assert!(
        dot.abs() > 0.5,
        "Center normals should be similar (up to sign), dot={}",
        dot
    );

    Ok(())
}

#[test]
fn test_normal_map_consistency() -> Result<(), Box<dyn Error>> {
    let Some(ctx) = require_gpu()? else {
        return Ok(());
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

/// Index into a dense `(gx, gy, gz)` field laid out `z-major` as the function returns.
fn field_at(field: &[f32], gx: u32, gy: u32, x: u32, y: u32, z: u32) -> f32 {
    field[((z * gy + y) * gx + x) as usize]
}

#[test]
fn distance_field_is_signed_surface_distance_not_closest_vertex() {
    // Unit triangle in the xy-plane, winding CCW → normal +z.
    // Old body returned unsigned distance to the nearest *vertex*; a sample
    // above the centroid must now report ≈ +height (surface) and flip sign
    // below the plane — neither of which a closest-vertex field can do.
    let vertices = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        Point3::new(0.0, 2.0, 0.0),
    ];
    let faces = [[0u32, 1, 2]];

    // Centroid of the triangle is at (2/3, 2/3, 0). Put that on a grid cell
    // and sample one voxel above and one below.
    let voxel = 0.5_f32;
    let gx = 5u32;
    let gy = 5u32;
    let gz = 5u32;
    // Origin so cell (1,1,2) sits at world (2/3, 2/3, 0):
    //   origin + i * voxel = target  ⇒  origin = target - i * voxel
    let cx = 2.0 / 3.0;
    let cy = 2.0 / 3.0;
    let origin = Point3::new(cx - 1.0 * voxel, cy - 1.0 * voxel, 0.0 - 2.0 * voxel);

    let field = cv_hal::gpu_kernels::raycasting_gpu::compute_distance_field(
        &vertices,
        &faces,
        origin,
        (gx, gy, gz),
        voxel,
    )
    .expect("distance field");
    assert_eq!(field.len(), (gx * gy * gz) as usize);

    let on_surface = field_at(&field, gx, gy, 1, 1, 2);
    let above = field_at(&field, gx, gy, 1, 1, 3); // +voxel in z
    let below = field_at(&field, gx, gy, 1, 1, 1); // -voxel in z

    assert!(
        on_surface.abs() < 1e-4,
        "centroid cell should be on the surface, got {on_surface}"
    );
    assert!(
        (above - voxel).abs() < 1e-4,
        "one voxel above must be +{voxel} (surface distance), got {above}"
    );
    assert!(
        (below + voxel).abs() < 1e-4,
        "one voxel below must be -{voxel} (signed), got {below}"
    );

    // Closest-vertex would have reported distance to (0,0,0)/(2,0,0)/(0,2,0)
    // from the raised centroid ≈ sqrt((2/3)^2 + (2/3)^2 + voxel^2) ≈ 1.05,
    // not `voxel`. Pin that the old answer is gone.
    let vertex_lie = (cx * cx + cy * cy + voxel * voxel).sqrt();
    assert!(
        (above - vertex_lie).abs() > 0.2,
        "must not still be returning closest-vertex distance ({vertex_lie}), got {above}"
    );
}

#[test]
fn distance_field_empty_inputs_return_empty() {
    let empty: Vec<Point3<f32>> = vec![];
    let faces: Vec<[u32; 3]> = vec![];
    let out = cv_hal::gpu_kernels::raycasting_gpu::compute_distance_field(
        &empty,
        &faces,
        Point3::origin(),
        (4, 4, 4),
        0.1,
    )
    .unwrap();
    assert!(out.is_empty());
}
