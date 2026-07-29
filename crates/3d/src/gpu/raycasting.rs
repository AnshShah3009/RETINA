use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::spatial::bvh::Bvh;

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
                let e1 = v[face[1]] - v[face[0]];
                let e2 = v[face[2]] - v[face[0]];
                let mut n = e1.cross(&e2);
                let len = n.norm();
                if len > 1e-9 {
                    n /= len;
                }
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
                            let e1 = v1 - v0;
                            let e2 = v2 - v0;
                            let mut n = e1.cross(&e2);
                            let len = n.norm();
                            if len > 1e-9 {
                                n /= len;
                            }
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
    if a.abs() < 1e-9 {
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
