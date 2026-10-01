use nalgebra::{Point3, Vector3};

#[test]
fn probe_aa_icp_sign_on_wellconditioned_target() {
    use cv_3d::gpu::registration::icp_point_to_plane;
    // Non-planar (bumpy) target -> AtA should be well conditioned.
    // target = z = f(x,y) bumpy surface; normals analytic.
    let f = |x: f32, y: f32| -> f32 { 0.3*(x*0.5).sin() + 0.25*(y*0.7).cos() + 0.1*(x*y*0.2).sin() };
    let nrm = |x: f32, y: f32| -> Vector3<f32> {
        let h = 1e-3;
        let dzdx = (f(x+h,y)-f(x-h,y))/(2.0*h);
        let dzdy = (f(x,y+h)-f(x,y-h))/(2.0*h);
        Vector3::new(-dzdx, -dzdy, 1.0).normalize()
    };
    let mut tgt = Vec::new(); let mut normals = Vec::new();
    for i in 0..40 { for j in 0..40 {
        let x = i as f32 * 0.1 - 2.0; let y = j as f32 * 0.1 - 2.0;
        tgt.push(Point3::new(x, y, f(x,y)));
        normals.push(nrm(x,y));
    }}
    println!("target points: {}", tgt.len());
    // source = target translated by a KNOWN small vector
    let shift = Vector3::new(0.03, -0.02, 0.01);
    let src: Vec<Point3<f32>> = tgt.iter().map(|p| Point3::from(p.coords + shift)).collect();

    for iters in [1usize, 2, 5, 20] {
        match icp_point_to_plane(&src, &tgt, &normals, 1.0, iters) {
            Ok(m) => {
                let t = m.fixed_view::<3,1>(0,3).into_owned();
                let err = src.iter().map(|p| {
                    let tp = m.transform_point(p);
                    tgt.iter().map(|q| (tp - q).norm()).fold(f32::MAX, f32::min)
                }).fold(0f32, f32::max);
                println!("PROBE AA iters={:2}: t=({:+.4},{:+.4},{:+.4})  worst_resid={:.5}",
                    iters, t.x, t.y, t.z, err);
            }
            Err(e) => println!("PROBE AA iters={:2}: ERR {}", iters, e),
        }
    }
    println!("   ground truth shift = ({:+.3},{:+.3},{:+.3})  -> ICP should recover ~ -shift", -shift.x, -shift.y, -shift.z);
    // Cross-check: same setup through the odometry point-to-plane solver path
    use cv_3d::tsdf::CameraIntrinsics;
    println!("(odometry path compared separately)");
}

#[test]
fn probe_bb_odometry_recovers_known_shift() {
    use cv_3d::odometry::{compute_rgbd_odometry, OdometryMethod};
    use cv_3d::tsdf::CameraIntrinsics;
    let w = 160usize; let h = 120usize;
    let intr = CameraIntrinsics::new(120.0, 120.0, w as f32/2.0, h as f32/2.0, w as u32, h as u32);
    // bumpy fronto-parallel surface at z~2
    let f = |u: f32, v: f32| -> f32 {
        let x = (u - intr.cx)/intr.fx; let y = (v - intr.cy)/intr.fy;
        2.0 + 0.25*(x*4.0).sin()*x.abs().min(1.0) + 0.25*(y*4.0).cos()*y.abs().min(1.0)
    };
    let mut d0 = vec![0f32; w*h];
    for v in 0..h { for u in 0..w { d0[v*w+u] = f(u as f32, v as f32); } }
    // target = same surface, camera translated +x by 0.05 and +z 0.02
    let mut d1 = vec![0f32; w*h];
    for v in 0..h { for u in 0..w {
        let x = ((u as f32) - intr.cx)/intr.fx; let y = ((v as f32) - intr.cy)/intr.fy;
        let z = f(u as f32, v as f32);
        // shift camera: point (x,y,z) appears at u' where x' = x - dx/z ...
        let dx = 0.02; let dz = 0.01;
        let xr = x - dx/z; let zr = z + dz;
        let uu = (xr*intr.fx*zr + intr.cx).round();
        if uu >= 0.0 && uu < w as f32 { d1[v*w + uu as usize] = zr; }
    }}
    let res = compute_rgbd_odometry(&d0, &d1, None, None, &intr, w, h, OdometryMethod::PointToPlane);
    match res {
        Some(r) => {
            let t = r.transformation.fixed_view::<3,1>(0,3).into_owned();
            println!("PROBE BB: fitness={:.4} rmse={:.6} t=({:+.5},{:+.5},{:+.5})", r.fitness, r.inlier_rmse, t.x, t.y, t.z);
        }
        None => println!("PROBE BB: None"),
    }
}

#[test]
fn probe_cc_downsample_depth_oob() {
    // downsample_depth is private; exercise through odometry with a mismatched slice.
    use cv_3d::odometry::{compute_rgbd_odometry, OdometryMethod};
    use cv_3d::tsdf::CameraIntrinsics;
    let w = 64usize; let h = 64usize;
    let intr = CameraIntrinsics::new(50.0,50.0,32.0,32.0,w as u32,h as u32);
    // depth slices much SHORTER than w*h
    let short = vec![1.0f32; 10];
    let r = std::panic::catch_unwind(|| {
        compute_rgbd_odometry(&short, &short, None, None, &intr, w, h, OdometryMethod::PointToPlane)
    });
    println!("PROBE CC: odometry with depth len=10 but w*h=4096 -> {}", if r.is_ok() {"ok"} else {"PANIC"});
    // TSDF integrate with short depth
    let mut vol = cv_3d::TSDFVolume::new(0.05, 0.1);
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        vol.integrate_ctx(&short, None, &intr, &nalgebra::Matrix4::identity(), w, h,
            &cv_runtime_unwrap());
    }));
    println!("PROBE CC2: TSDFVolume::integrate_ctx with short depth -> {}", if r2.is_ok() {"ok"} else {"PANIC"});
    // TSDF with short COLOR image
    let full = vec![1.0f32; w*h];
    let short_c = vec![nalgebra::Vector3::<u8>::new(1,2,3); 5];
    let mut vol2 = cv_3d::TSDFVolume::new(0.05, 0.1);
    let r3 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        vol2.integrate_ctx(&full, Some(&short_c), &intr, &nalgebra::Matrix4::identity(), w, h,
            &cv_runtime_unwrap());
    }));
    println!("PROBE CC3: TSDFVolume::integrate_ctx with short COLOR -> {}", if r3.is_ok() {"ok"} else {"PANIC"});
}

fn cv_runtime_unwrap() -> cv_runtime::orchestrator::RuntimeRunner {
    cv_runtime::best_runner().unwrap_or_else(|_| {
        cv_runtime::orchestrator::RuntimeRunner::Sync(cv_hal::DeviceId(0))
    })
}

#[test]
fn probe_dd_filters_voxel_short_normals() {
    let pts: Vec<Point3<f64>> = vec![Point3::new(0.0,0.0,0.0), Point3::new(1.0,0.0,0.0)];
    let r = std::panic::catch_unwind(|| {
        cv_3d::filters::voxel_downsample(&pts, Some(&vec![Vector3::new(0.0,0.0,1.0)]), None, 2.0)
    });
    println!("PROBE DD: filters::voxel_downsample(points=2, normals=1) -> {}", if r.is_ok() {"ok"} else {"PANIC"});
}

#[test]
fn probe_ee_hashgrid_radius_zero_and_negative() {
    use cv_3d::spatial::HashGrid;
    let pts = vec![Point3::new(0.0,0.0,0.0), Point3::new(0.1,0.0,0.0)];
    let g = HashGrid::build(&pts, 1.0);
    println!("PROBE EE: radius=0 -> {} (the query point itself: dist 0 <= 0)", g.radius_search(&Point3::new(0.0,0.0,0.0), 0.0).len());
    let r = std::panic::catch_unwind(|| g.radius_search(&Point3::new(0.0,0.0,0.0), -1.0).len());
    println!("PROBE EE: radius=-1 -> {}", if r.is_ok() {"ok"} else {"PANIC"});
    // empty grid
    let e = HashGrid::build(&[], 1.0);
    println!("PROBE EE: empty grid len={} is_empty={}", e.len(), e.is_empty());
    println!("PROBE EE: empty radius_search -> {:?}", e.radius_search(&Point3::new(0.0,0.0,0.0), 1.0).len());
    println!("PROBE EE: empty nearest -> {:?}", e.nearest(&Point3::new(0.0,0.0,0.0), 1.0).is_some());
}

#[test]
fn probe_ff_kdtree_empty() {
    use cv_3d::spatial::KDTree;
    let t: KDTree<usize> = KDTree::new();
    println!("PROBE FF: empty nearest = {:?}", t.nearest_neighbor(&Point3::new(0.0,0.0,0.0)).is_some());
    println!("PROBE FF: empty radius = {}", t.search_radius(&Point3::new(0.0,0.0,0.0), 1.0).len());
    println!("PROBE FF: empty knn = {}", t.k_nearest_neighbors(&Point3::new(0.0,0.0,0.0), 5).len());
    // duplicate points -> ties
    let mut items: Vec<(Point3<f32>, usize)> = (0..10).map(|i| (Point3::new(0.0,0.0,0.0), i)).collect();
    let t2 = KDTree::build(&mut items);
    let knn = t2.k_nearest_neighbors(&Point3::new(0.0,0.0,0.0), 10);
    println!("PROBE FF: 10 identical pts, k=10 -> {} results (all dist 0)", knn.len());
}
