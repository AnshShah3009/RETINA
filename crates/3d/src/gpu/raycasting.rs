use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::spatial::bvh::Bvh;

/// Angular tolerance for "this ray is parallel to this triangle", as a
/// dimensionless bound on `|cos(angle between ray and triangle normal)|`.
///
/// The same value, for the same reason, as `spatial::bvh::PARALLEL_EPS`.
const PARALLEL_EPS: f32 = 1e-9;

/// Unit-length surface normal of the triangle `(v0, v1, v2)`.
///
/// Returns `None` for a zero-area triangle, which has no normal.
///
/// The shared implementation of this normalisation matters: three copies of
/// this routine existed (`raycasting::cast_ray_mesh_bvh`,
/// `gpu::raycasting::cast_rays_with_bvh` and `cast_rays_brute`), and the two in
/// this file guarded it with `len > 1e-9` on the **unnormalised** cross product.
/// That length is the triangle area and scales as L^2, so the guard was a size
/// test, not a normalisation guard.
/// Returns the zero vector for a zero-area triangle, which has no normal.
fn triangle_normal(v0: &Point3<f32>, v1: &Point3<f32>, v2: &Point3<f32>) -> Vector3<f32> {
    let n = (v1 - v0).cross(&(v2 - v0));
    let len = n.norm();
    if len > 0.0 {
        n / len
    } else {
        Vector3::zeros()
    }
}

/// BVH-accelerated ray-mesh intersection — O(rays * log(triangles)).
///
/// Builds a BVH on first call. For repeated queries against the same mesh,
/// use [`cast_rays_with_bvh`] to reuse the BVH.
#[allow(clippy::type_complexity)]
pub fn cast_rays(
    ro: &[Point3<f32>],
    rd: &[Vector3<f32>],
    v: &[Point3<f32>],
    f: &[[usize; 3]],
) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
    let bvh = Bvh::build(v, f);
    cast_rays_with_bvh(ro, rd, v, f, &bvh)
}

/// Ray-mesh intersection using a pre-built BVH.
#[allow(clippy::type_complexity)]
pub fn cast_rays_with_bvh(
    ro: &[Point3<f32>],
    rd: &[Vector3<f32>],
    v: &[Point3<f32>],
    f: &[[usize; 3]],
    bvh: &Bvh,
) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
    let results: Vec<_> = ro
        .par_iter()
        .zip(rd.par_iter())
        .map(|(origin, dir)| {
            bvh.intersect_ray(origin, dir, v, f).map(|(t, fi, _u, _v)| {
                let hit = Point3::from(origin.coords + dir * t);
                let face = &f[fi];
                let n = triangle_normal(&v[face[0]], &v[face[1]], &v[face[2]]);
                (t, hit, n)
            })
        })
        .collect();
    Ok(results)
}

/// Brute-force ray-mesh intersection — O(rays * triangles).
/// Kept for correctness comparison and small meshes.
#[allow(clippy::type_complexity)]
pub fn cast_rays_brute(
    ro: &[Point3<f32>],
    rd: &[Vector3<f32>],
    v: &[Point3<f32>],
    f: &[[usize; 3]],
) -> Result<Vec<Option<(f32, Point3<f32>, Vector3<f32>)>>, String> {
    let results: Vec<_> = ro
        .par_iter()
        .zip(rd.par_iter())
        .map(|(origin, dir)| {
            let mut best: Option<(f32, Point3<f32>, Vector3<f32>)> = None;
            for face in f {
                let v0 = v[face[0]];
                let v1 = v[face[1]];
                let v2 = v[face[2]];
                if let Some((t, _u, _v)) = moller_trumbore(&origin.coords, dir, &v0, &v1, &v2) {
                    if t > 1e-6 {
                        let replace = match best {
                            None => true,
                            Some((bt, _, _)) => t < bt,
                        };
                        if replace {
                            let hit = Point3::from(origin.coords + dir * t);
                            let n = triangle_normal(&v0, &v1, &v2);
                            best = Some((t, hit, n));
                        }
                    }
                }
            }
            best
        })
        .collect();
    Ok(results)
}

/// Möller-Trumbore ray-triangle intersection.
/// Returns `Some((t, u, v))` if ray `origin + t*dir` hits the triangle.
fn moller_trumbore(
    origin: &Vector3<f32>,
    dir: &Vector3<f32>,
    v0: &Point3<f32>,
    v1: &Point3<f32>,
    v2: &Point3<f32>,
) -> Option<(f32, f32, f32)> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = dir.cross(&e2);
    let a = e1.dot(&h);
    // `a` is twice the *projected* triangle area, so it scales as L^2 - the
    // square of the edge length. Comparing it against a fixed constant is
    // therefore a size test, not the angular test it looks like, and the size it
    // rejects moves with the world units the mesh happens to be stored in.
    //
    // This is the *third* copy of this defect. It was found and fixed in
    // `spatial::bvh::moller_trumbore` (the BVH path) and in
    // `raycasting::ray_triangle_intersection` (the CPU inside/outside test); the
    // brute-force reference implementation below kept the original absolute
    // threshold. That matters twice over, because `cast_rays_brute` is documented
    // as the correctness comparison for the BVH path: the reference disagreed
    // with the thing it was checking, at exactly the scale where the difference
    // shows.
    //
    // Measured, equilateral triangle of edge `s` hit dead-on:
    // s = 1e-3 gives |a| = 8.66e-7 (above the old 1e-9 threshold, still a hit),
    // s = 1e-5 gives |a| = 8.66e-11, below it - so a millimetre-unit mesh lost
    // every ray here while the BVH path found them all. Dividing by |e1||e2|
    // makes the test the dimensionless bound on the angle between the ray and
    // the triangle normal that it was always meant to be, leaving the angular
    // sharpness (`PARALLEL_EPS`) unchanged and removing only the scale
    // dependence.
    let denom = e1.norm() * e2.norm();
    if denom == 0.0 || a.abs() < PARALLEL_EPS * denom {
        return None; // parallel
    }
    let f = 1.0 / a;
    let s = Point3::from(*origin) - v0;
    let u = f * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(&e1);
    let v = f * dir.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(&q);
    Some((t, u, v))
}
