use nalgebra::{Matrix4, Point3, Vector3};
use rayon::prelude::*;

/// Simple ICP point-to-plane registration.
///
/// For full-featured ICP (multi-scale, robust kernels, colored),
/// use the `cv-registration` crate instead.
pub fn icp_point_to_plane(
    source: &[Point3<f32>],
    target: &[Point3<f32>],
    target_normals: &[Vector3<f32>],
    max_dist: f32,
    max_iters: usize,
) -> Result<Matrix4<f32>, String> {
    use crate::spatial::KDTree;

    if target.is_empty() || source.is_empty() || target_normals.len() != target.len() {
        return Err("Invalid input sizes".to_string());
    }

    let mut items: Vec<(Point3<f32>, usize)> =
        target.iter().copied().zip(0..target.len()).collect();
    let tree = KDTree::build(&mut items);
    let mut transform = Matrix4::identity();
    let max_dist_sq = max_dist * max_dist;

    for _ in 0..max_iters {
        // Find correspondences and build linear system (6x6) — parallel reduction
        let (ata, atb, n_corr) = source
            .par_iter()
            .fold(
                || {
                    (
                        nalgebra::Matrix6::<f64>::zeros(),
                        nalgebra::Vector6::<f64>::zeros(),
                        0usize,
                    )
                },
                |(mut ata, mut atb, mut count), sp| {
                    let tp = transform.transform_point(sp);
                    if let Some((closest, idx, dist_sq)) = tree.nearest_neighbor(&tp) {
                        if dist_sq <= max_dist_sq {
                            let n = nalgebra::Vector3::new(
                                target_normals[idx].x as f64,
                                target_normals[idx].y as f64,
                                target_normals[idx].z as f64,
                            );
                            let p =
                                nalgebra::Vector3::new(tp.x as f64, tp.y as f64, tp.z as f64);
                            let q = nalgebra::Vector3::new(
                                closest.x as f64,
                                closest.y as f64,
                                closest.z as f64,
                            );
                            let d = p - q;
                            let cross = p.cross(&n);
                            let row = nalgebra::Vector6::new(
                                cross.x, cross.y, cross.z, n.x, n.y, n.z,
                            );
                            let rhs = -n.dot(&d);
                            ata += row * row.transpose();
                            atb += row * rhs;
                            count += 1;
                        }
                    }
                    (ata, atb, count)
                },
            )
            .reduce(
                || {
                    (
                        nalgebra::Matrix6::<f64>::zeros(),
                        nalgebra::Vector6::<f64>::zeros(),
                        0usize,
                    )
                },
                |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
            );

        if n_corr < 6 {
            return Err(format!("Too few correspondences: {n_corr}"));
        }

        // Solve 6x6 system
        let x = ata
            .lu()
            .solve(&atb)
            .ok_or("Failed to solve ICP linear system")?;

        // Build incremental transform from twist vector
        let (a, b, g) = (x[0] as f32, x[1] as f32, x[2] as f32);
        let (tx, ty, tz) = (x[3] as f32, x[4] as f32, x[5] as f32);

        let mut inc = Matrix4::identity();
        inc[(0, 1)] = -g;
        inc[(0, 2)] = b;
        inc[(1, 0)] = g;
        inc[(1, 2)] = -a;
        inc[(2, 0)] = -b;
        inc[(2, 1)] = a;
        inc[(0, 3)] = tx;
        inc[(1, 3)] = ty;
        inc[(2, 3)] = tz;

        transform = inc * transform;

        // Convergence check
        let delta = x.norm();
        if delta < 1e-6 {
            break;
        }
    }

    Ok(transform)
}
