//! A fused TSDF must mesh back to the surface that was integrated - where it
//! is, and nowhere else.
//!
//! Three defects in `TSDFVolume`, each of which moved or tore the mesh of a
//! plain flat wall:
//!
//! 1. **Sign convention.** Untouched voxels start at `+1` ("empty"), but
//!    integration wrote `sdf = z_voxel - depth`, i.e. *negative* in observed
//!    free space. And marching cubes did not consult weights. So wherever the
//!    observed band met never-observed voxels there was a sign change, and a
//!    phantom surface was emitted there.
//! 2. **Depth treated as ray range.** The march went `depth +/- trunc` *along
//!    the unit ray*, but `depth` is a z value. Off axis the true surface sits at
//!    range `depth / cos(theta)`, so the band stopped short of `z = depth`.
//! 3. **Block seams.** `extract_mesh` iterated cells `0..BLOCK_SIZE - 1` inside
//!    each block, so a cell whose corners straddle two blocks was never meshed:
//!    a one-voxel crack every eight voxels.
//!
//! Measured before the fix on the fixture below (plane at z = 1, voxel 2 cm,
//! truncation 6 cm, fx = 100, 64x64): 8153 of 12921 vertices were more than
//! 3 cm off the plane, at z 0.83 - 0.94.

use cv_3d::tsdf::{TSDFVolume, Triangle};
use cv_core::CameraIntrinsicsF32 as CameraIntrinsics;
use nalgebra::{Matrix4, Point3};

const VOXEL: f32 = 0.02;
const TRUNC: f32 = 0.06;
const W: usize = 64;
const H: usize = 64;
const FX: f32 = 100.0;

/// Integrate a fronto-parallel plane at camera depth `plane_z`, seen from a
/// camera whose camera-to-world pose is `pose`, and mesh it.
fn mesh_plane(plane_z: f32, pose: &Matrix4<f32>) -> Vec<Triangle> {
    mesh_plane_fx(plane_z, pose, FX)
}

fn mesh_plane_fx(plane_z: f32, pose: &Matrix4<f32>, fx: f32) -> Vec<Triangle> {
    let mut vol = TSDFVolume::new(VOXEL, TRUNC).with_depth_scale(1.0);
    let k = CameraIntrinsics::new(fx, fx, 32.0, 32.0, W as u32, H as u32);
    let depth = vec![plane_z; W * H];
    vol.integrate(&depth, None, &k, pose, W, H)
        .expect("a well-formed frame must integrate");
    vol.extract_mesh()
}

/// Every vertex's distance from the world plane `z = z0`.
fn worst_offset(tris: &[Triangle], z0: f32) -> (usize, usize, f32) {
    let mut off = 0usize;
    let mut total = 0usize;
    let mut worst = 0.0f32;
    for t in tris {
        for p in &t.vertices {
            total += 1;
            let d = (p.z - z0).abs();
            worst = worst.max(d);
            if d > 0.5 * VOXEL {
                off += 1;
            }
        }
    }
    (off, total, worst)
}

fn assert_on_plane(tris: &[Triangle], z0: f32, label: &str) {
    assert!(
        tris.len() > 1000,
        "{label}: only {} triangles - the plane was not meshed",
        tris.len()
    );
    let (off, total, worst) = worst_offset(tris, z0);
    assert_eq!(
        off, 0,
        "{label}: {off} of {total} vertices are more than half a voxel off the \
         plane z = {z0} (worst {worst:.4} m). Phantom surfaces where the observed \
         band meets unobserved voxels, or a band that never reached the surface."
    );
}

#[test]
fn flat_plane_meshes_onto_the_plane() {
    let tris = mesh_plane(1.0, &Matrix4::identity());
    assert_on_plane(&tris, 1.0, "identity");
}

#[test]
fn face_normals_point_out_of_the_surface() {
    // Positive is free space, so the outward normal of a wall seen from the
    // origin points back at the camera, -z. Flipping the sign convention
    // without flipping the winding would turn every face inside out.
    let tris = mesh_plane(1.0, &Matrix4::identity());
    let inward = tris.iter().filter(|t| t.normals[0].z > -0.99).count();
    assert_eq!(
        inward,
        0,
        "{inward} of {} face normals do not point toward the camera",
        tris.len()
    );
}

#[test]
fn no_vertex_on_the_far_side_of_the_surface() {
    // The far side of a wall is never observed: nothing may be meshed there.
    let tris = mesh_plane(1.0, &Matrix4::identity());
    let behind = tris
        .iter()
        .flat_map(|t| t.vertices.iter())
        .filter(|p| p.z > 1.0 + 0.5 * VOXEL)
        .count();
    assert_eq!(behind, 0, "{behind} vertices behind the observed wall");
}

/// A wide field of view, where `depth` and range along the ray differ most.
///
/// With fx = 30 the corner pixel's ray is ~60 degrees off axis, so the surface
/// at z = 1 is at range ~1.8. Marching `depth +/- trunc` along the unit ray -
/// treating the z value as a range - stops ~0.8 m short of it, and the edge of
/// the image never gets a surface at all.
#[test]
fn wide_field_of_view_reaches_the_plane_off_axis() {
    let tris = mesh_plane_fx(1.0, &Matrix4::identity(), 30.0);
    assert_on_plane(&tris, 1.0, "fx = 30");
    // Pixel 0 backprojects to x = -32 / 30 = -1.07 at z = 1; the mesh must
    // reach most of the way out there.
    let reach = tris
        .iter()
        .flat_map(|t| t.vertices.iter())
        .map(|p| p.x.abs().min(p.y.abs()))
        .fold(0.0f32, f32::max);
    assert!(
        reach > 0.9,
        "the mesh only reaches {reach:.3} m off axis in both x and y; the image \
         corners see the plane out to ~1.07 m"
    );
}

#[test]
fn translated_camera_meshes_onto_the_plane() {
    // Camera-to-world: the camera sits at (0.13, -0.07, 0.31), so the plane at
    // camera depth 0.9 is the world plane z = 1.21. Non-zero, non-aligned
    // offsets so block seams fall at arbitrary places on the image.
    let mut pose = Matrix4::identity();
    pose[(0, 3)] = 0.13;
    pose[(1, 3)] = -0.07;
    pose[(2, 3)] = 0.31;
    let tris = mesh_plane(0.9, &pose);
    assert_on_plane(&tris, 1.21, "translated");
}

/// Is `(x, y)` inside the xy-projection of `t`?
fn covers(t: &Triangle, x: f32, y: f32) -> bool {
    let [a, b, c] = t.vertices;
    let s = |p: Point3<f32>, q: Point3<f32>| (q.x - p.x) * (y - p.y) - (q.y - p.y) * (x - p.x);
    let (d0, d1, d2) = (s(a, b), s(b, c), s(c, a));
    let eps = 1e-7;
    let neg = d0 < -eps || d1 < -eps || d2 < -eps;
    let pos = d0 > eps || d1 > eps || d2 > eps;
    !(neg && pos)
}

#[test]
fn no_crack_at_block_seams() {
    let tris = mesh_plane(1.0, &Matrix4::identity());
    assert!(!tris.is_empty());

    // Bucket triangles by the voxel column their centroid sits in, so each
    // probe only tests its neighbourhood.
    let mut buckets: std::collections::HashMap<(i32, i32), Vec<&Triangle>> =
        std::collections::HashMap::new();
    for t in &tris {
        let cx = (t.vertices[0].x + t.vertices[1].x + t.vertices[2].x) / 3.0;
        let cy = (t.vertices[0].y + t.vertices[1].y + t.vertices[2].y) / 3.0;
        let key = ((cx / VOXEL).floor() as i32, (cy / VOXEL).floor() as i32);
        buckets.entry(key).or_default().push(t);
    }

    // The frustum at z = 1 spans +/- 0.32 m. Probe the inner +/- 0.2 m on a
    // grid finer than a voxel; blocks are 8 voxels = 16 cm, so seams at
    // x, y in {-0.16, 0, 0.16} fall inside the probed square.
    let mut holes = Vec::new();
    let n = 80;
    for i in 0..=n {
        for j in 0..=n {
            let x = -0.2 + 0.4 * i as f32 / n as f32 + 1e-4;
            let y = -0.2 + 0.4 * j as f32 / n as f32 + 1e-4;
            let (kx, ky) = ((x / VOXEL).floor() as i32, (y / VOXEL).floor() as i32);
            let mut hit = false;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(ts) = buckets.get(&(kx + dx, ky + dy))
                        && ts.iter().any(|t| covers(t, x, y))
                    {
                        hit = true;
                        break 'search;
                    }
                }
            }
            if !hit {
                holes.push((x, y));
            }
        }
    }
    assert!(
        holes.is_empty(),
        "{} of {} probe points over the plane are not covered by any triangle; \
         first few: {:?}. A crack along a block seam every 8 voxels.",
        holes.len(),
        (n + 1) * (n + 1),
        &holes[..holes.len().min(5)]
    );
}

#[test]
fn point_cloud_sits_on_the_plane_with_outward_normals() {
    let mut vol = TSDFVolume::new(VOXEL, TRUNC).with_depth_scale(1.0);
    let k = CameraIntrinsics::new(FX, FX, 32.0, 32.0, W as u32, H as u32);
    // Off the voxel lattice on purpose: `extract_point_cloud` keeps voxels with
    // |tsdf| < 0.1, i.e. centres within 0.1 * trunc = 6 mm of the surface, and
    // at z = 1.005 that is exactly the voxel centred on 1.01.
    let plane = 1.005;
    let depth = vec![plane; W * H];
    vol.integrate(&depth, None, &k, &Matrix4::identity(), W, H)
        .unwrap();
    let pts = vol.extract_point_cloud();
    assert!(!pts.is_empty());
    for (p, _, _) in &pts {
        assert!(
            (p.z - plane).abs() <= 0.5 * VOXEL + 1e-4,
            "surface point at z = {} is not adjacent to the plane",
            p.z
        );
    }
    // The gradient of a positive-outside TSDF points out of the surface, i.e.
    // back toward the camera (-z). Checked where the stencil is interior.
    let toward_camera = pts
        .iter()
        .filter(|(p, _, _)| p.x.abs() < 0.15 && p.y.abs() < 0.15)
        .filter(|(_, n, _)| n.z < -0.5)
        .count();
    let interior = pts
        .iter()
        .filter(|(p, _, _)| p.x.abs() < 0.15 && p.y.abs() < 0.15)
        .count();
    assert!(interior > 0);
    assert!(
        toward_camera * 10 >= interior * 9,
        "only {toward_camera} of {interior} interior surface normals face the \
         camera"
    );
}
