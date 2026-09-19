//! Tests for all performance optimizations added in the optimization session.
//! Covers: balanced KDTree, BVH, HashGrid, ICP, raycasting, Poisson, MC tables.

use cv_3d::mesh::reconstruction::poisson_reconstruction;
use cv_3d::spatial::bvh::Bvh;
use cv_3d::spatial::hash_grid::HashGrid;
use cv_3d::spatial::KDTree;
use cv_core::PointCloud;
use nalgebra::{Point3, Vector3};

fn sphere_cloud(n: usize, radius: f32) -> PointCloud {
    let golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut pts = Vec::with_capacity(n);
    let mut nrm = Vec::with_capacity(n);
    for i in 0..n {
        let theta = 2.0 * std::f32::consts::PI * (i as f32) / golden;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / n as f32).acos();
        let (sp, cp) = (phi.sin(), phi.cos());
        let (st, ct) = (theta.sin(), theta.cos());
        let p = Point3::new(radius * sp * ct, radius * sp * st, radius * cp);
        pts.push(p);
        nrm.push(Vector3::new(p.x, p.y, p.z).normalize());
    }
    PointCloud::new(pts).with_normals(nrm).unwrap()
}

// ── KDTree balanced build ────────────────────────────────────────────────────

#[test]
fn test_kdtree_build_matches_insert() {
    let cloud = sphere_cloud(500, 1.0);
    let query = Point3::new(0.5, 0.5, 0.5);

    // Build via balanced construction
    let mut items: Vec<_> = cloud
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();
    let tree_built = KDTree::build(&mut items);

    // Build via incremental insert
    let mut tree_insert = KDTree::new();
    for (i, pt) in cloud.points.iter().enumerate() {
        tree_insert.insert(*pt, i);
    }

    // Both should find the same nearest neighbor
    let (_, idx_built, dist_built) = tree_built.nearest_neighbor(&query).unwrap();
    let (_, idx_insert, dist_insert) = tree_insert.nearest_neighbor(&query).unwrap();
    assert_eq!(idx_built, idx_insert);
    assert!((dist_built - dist_insert).abs() < 1e-6);
}

#[test]
fn test_kdtree_knn_correct_count() {
    let cloud = sphere_cloud(1000, 1.0);
    let mut items: Vec<_> = cloud
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();
    let tree = KDTree::build(&mut items);

    let results = tree.k_nearest_neighbors(&Point3::new(0.0, 0.0, 0.0), 10);
    assert_eq!(results.len(), 10);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].2 >= results[i - 1].2);
    }
}

#[test]
fn test_kdtree_radius_search() {
    let cloud = sphere_cloud(1000, 1.0);
    let mut items: Vec<_> = cloud
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();
    let tree = KDTree::build(&mut items);

    let results = tree.search_radius(&Point3::new(1.0, 0.0, 0.0), 0.3);
    // All results should be within radius
    for (pt, _, dist_sq) in &results {
        assert!(dist_sq.sqrt() <= 0.3 + 1e-6);
        let d = (pt - Point3::new(1.0, 0.0, 0.0)).norm();
        assert!(d <= 0.3 + 1e-6);
    }
}

// ── BVH ─────────────────────────────────────────────────────────────────────

#[test]
fn test_bvh_ray_hit() {
    let vertices = vec![
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];
    let bvh = Bvh::build(&vertices, &faces);

    // Ray from z=-2 toward z=0 should hit
    let origin = Point3::new(0.0, 0.0, -2.0);
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let hit = bvh.intersect_ray(&origin, &dir, &vertices, &faces);
    assert!(hit.is_some());
    let (t, face_idx, _u, _v) = hit.unwrap();
    assert!((t - 2.0).abs() < 1e-4);
    assert_eq!(face_idx, 0);
}

#[test]
fn test_bvh_ray_miss() {
    let vertices = vec![
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let faces = vec![[0, 1, 2]];
    let bvh = Bvh::build(&vertices, &faces);

    // Ray parallel to triangle should miss
    let origin = Point3::new(0.0, 0.0, -2.0);
    let dir = Vector3::new(1.0, 0.0, 0.0);
    assert!(bvh
        .intersect_ray(&origin, &dir, &vertices, &faces)
        .is_none());
}

#[test]
fn test_bvh_matches_brute_force() {
    // Build a mesh from Poisson and compare BVH vs brute-force
    let cloud = sphere_cloud(2000, 1.0);
    let mesh = poisson_reconstruction(&cloud, 5, 1.0).unwrap();

    let bvh = Bvh::build(&mesh.vertices, &mesh.faces);

    // Cast rays from outside toward center — these should all hit the sphere mesh
    for i in 0..20 {
        let theta = i as f32 * 0.31;
        let phi = 0.2 + i as f32 * 0.1;
        let origin = Point3::new(
            3.0 * phi.sin() * theta.cos(),
            3.0 * phi.sin() * theta.sin(),
            3.0 * phi.cos(),
        );
        let dir = (Point3::origin() - origin).normalize();

        let bvh_hit = bvh.intersect_ray(&origin, &dir, &mesh.vertices, &mesh.faces);

        // Brute-force
        let mut brute_best: Option<f32> = None;
        for face in &mesh.faces {
            let v0 = mesh.vertices[face[0]];
            let v1 = mesh.vertices[face[1]];
            let v2 = mesh.vertices[face[2]];
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let h = dir.cross(&e2);
            let a = e1.dot(&h);
            if a.abs() < 1e-9 {
                continue;
            }
            let f = 1.0 / a;
            let s = origin - v0;
            let u = f * s.dot(&h);
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(&e1);
            let v = f * dir.dot(&q);
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = f * e2.dot(&q);
            if t > 1e-6 {
                brute_best = Some(match brute_best {
                    None => t,
                    Some(bt) => bt.min(t),
                });
            }
        }

        // Both should agree
        match (bvh_hit, brute_best) {
            (Some((t_bvh, _, _, _)), Some(t_brute)) => {
                assert!(
                    (t_bvh - t_brute).abs() < 1e-3,
                    "Ray {}: BVH t={} vs brute t={}",
                    i,
                    t_bvh,
                    t_brute
                );
            }
            (None, None) => {} // both miss, ok
            _ => panic!("Ray {}: BVH and brute-force disagree on hit/miss", i),
        }
    }
}

// ── HashGrid ────────────────────────────────────────────────────────────────

#[test]
fn test_hashgrid_nearest_matches_kdtree() {
    let cloud = sphere_cloud(5000, 1.0);
    let query = Point3::new(0.7, 0.3, 0.5);

    let mut items: Vec<_> = cloud
        .points
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i))
        .collect();
    let tree = KDTree::build(&mut items);
    let hg = HashGrid::build(&cloud.points, 0.5);

    let (_, kd_idx, kd_dist) = tree.nearest_neighbor(&query).unwrap();
    let (hg_idx, _, hg_dist) = hg.nearest(&query, 0.5).unwrap();

    assert_eq!(kd_idx, hg_idx);
    assert!((kd_dist - hg_dist).abs() < 1e-5);
}

// ── Poisson SOR solver ──────────────────────────────────────────────────────

#[test]
fn test_poisson_produces_mesh() {
    let cloud = sphere_cloud(5000, 1.0);
    let mesh = poisson_reconstruction(&cloud, 5, 1.0);
    assert!(mesh.is_some(), "Poisson should produce a mesh");
    let mesh = mesh.unwrap();
    assert!(mesh.vertices.len() > 100, "Mesh should have vertices");
    assert!(mesh.faces.len() > 100, "Mesh should have faces");
}

// ── Marching cubes tables ───────────────────────────────────────────────────

#[test]
fn test_mc_tables_valid_edge_indices() {
    // All non-sentinel values in TRI_TABLE must be 0-11
    let tables = &cv_hal::gpu_kernels::marching_cubes_tables::TRI_TABLE;
    for (i, &val) in tables.iter().enumerate() {
        if val != -1 {
            assert!(
                (0..=11).contains(&val),
                "TRI_TABLE[{}] = {} — invalid edge index (must be 0-11 or -1)",
                i,
                val
            );
        }
    }
}

#[test]
fn test_mc_tables_tri_count_matches() {
    let tri_table = &cv_hal::gpu_kernels::marching_cubes_tables::TRI_TABLE;
    let tri_count = &cv_hal::gpu_kernels::marching_cubes_tables::TRI_COUNT;

    for case in 0..256 {
        let row = &tri_table[case * 16..case * 16 + 16];
        let mut count = 0u32;
        let mut i = 0;
        while i < 16 && row[i] != -1 {
            count += 1;
            i += 3;
        }
        assert_eq!(
            tri_count[case], count,
            "Case {}: TRI_COUNT={} but counted {} triangles",
            case, tri_count[case], count
        );
    }
}
