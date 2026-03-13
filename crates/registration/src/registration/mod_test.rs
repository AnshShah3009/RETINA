#[cfg(test)]
mod tests {
    use crate::registration::{
        registration_icp_point_to_plane, registration_icp_point_to_plane_ctx, GNCOptimizer,
        RobustLoss,
    };
    use cv_core::point_cloud::PointCloud;
    use cv_hal::compute::ComputeDevice;
    use cv_hal::cpu::CpuBackend;
    use nalgebra::{Matrix4, Point3, Vector3};

    #[test]
    fn test_icp_acceleration_parity() {
        let cpu = CpuBackend::new().unwrap();
        let device = ComputeDevice::Cpu(&cpu);

        let mut source = PointCloud::new(Vec::new());
        let mut target = PointCloud::new(Vec::new());

        // Create a simple plane with normals
        for y in 0..10 {
            for x in 0..10 {
                let p = Point3::new(x as f32 * 0.1, y as f32 * 0.1, 0.0);
                source.points.push(p);
                // Offset target slightly
                target.points.push(p + Vector3::new(0.02, 0.01, 0.0));
            }
        }
        target.normals = Some(vec![Vector3::new(0.0, 0.0, 1.0); 100]);

        let init = Matrix4::identity();
        let res_ctx =
            registration_icp_point_to_plane_ctx(&source, &target, 0.2, &init, 20, &device);

        match res_ctx {
            Ok(res) => {
                println!(
                    "ICP Result: fitness={}, rmse={}, iterations={}",
                    res.fitness, res.inlier_rmse, res.num_iterations
                );
                assert!(res.fitness > 0.9);
                assert!(res.inlier_rmse < 0.05);
            }
            Err(e) => {
                panic!("ICP failed to return a result: {}", e);
            }
        }
    }

    #[test]
    fn test_icp_identical_point_clouds() {
        let mut source = PointCloud::new(Vec::new());
        let mut target = PointCloud::new(Vec::new());

        // Create identical point clouds
        for y in 0..5 {
            for x in 0..5 {
                let p = Point3::new(x as f32, y as f32, 0.0);
                source.points.push(p);
                target.points.push(p);
            }
        }
        target.normals = Some(vec![Vector3::new(0.0, 0.0, 1.0); 25]);

        let init = Matrix4::identity();
        let result = registration_icp_point_to_plane(&source, &target, 10.0, &init, 10);

        assert!(result.is_some());
        if let Some(res) = result {
            // Identical point clouds should have perfect fitness
            assert!(res.fitness >= 0.99);
            assert!(res.inlier_rmse < 1e-5);
        }
    }

    #[test]
    fn test_icp_convergence_iterations() {
        let mut source = PointCloud::new(Vec::new());
        let mut target = PointCloud::new(Vec::new());

        // Create two slightly offset planes
        for y in 0..8 {
            for x in 0..8 {
                let p = Point3::new(x as f32 * 0.1, y as f32 * 0.1, 0.0);
                source.points.push(p);
                // Small translation offset
                target.points.push(p + Vector3::new(0.01, 0.01, 0.0));
            }
        }
        target.normals = Some(vec![Vector3::new(0.0, 0.0, 1.0); 64]);

        let init = Matrix4::identity();
        let result = registration_icp_point_to_plane(&source, &target, 1.0, &init, 50);

        assert!(result.is_some());
        if let Some(res) = result {
            // Should converge in reasonable iterations (< 50)
            assert!(res.num_iterations > 0);
            assert!(res.num_iterations <= 50);
            assert!(res.fitness > 0.8);
        }
    }

    #[test]
    fn test_icp_very_distant_point_clouds() {
        let source = PointCloud::new(vec![Point3::new(0.0, 0.0, 0.0)]);
        let target = PointCloud::new(vec![Point3::new(100.0, 100.0, 100.0)]);

        let init = Matrix4::identity();
        let result = registration_icp_point_to_plane(&source, &target, 1.0, &init, 10);

        // With very distant points and small distance threshold, either fails or has low fitness
        if let Some(res) = result {
            // If it returns a result, fitness should be very low
            assert!(res.fitness < 0.1);
        }
        // Either way is acceptable - the function handles this gracefully
    }

    #[test]
    fn test_icp_with_larger_distance_threshold() {
        let mut source = PointCloud::new(Vec::new());
        let mut target = PointCloud::new(Vec::new());

        // Create two offset planes with larger distance
        for y in 0..5 {
            for x in 0..5 {
                let p = Point3::new(x as f32, y as f32, 0.0);
                source.points.push(p);
                // Larger offset
                target.points.push(p + Vector3::new(0.5, 0.5, 0.0));
            }
        }
        target.normals = Some(vec![Vector3::new(0.0, 0.0, 1.0); 25]);

        let init = Matrix4::identity();
        let result = registration_icp_point_to_plane(&source, &target, 2.0, &init, 20);

        assert!(result.is_some());
        if let Some(res) = result {
            assert!(res.fitness > 0.9);
        }
    }

    // GNC Robust Loss Tests
    #[test]
    fn test_geman_mcclure_loss_evaluate() {
        let loss = RobustLoss::GemanMcClure { mu: 1.0 };

        // At r=0, loss should be 0
        let l0 = loss.evaluate(0.0);
        assert!((l0 - 0.0).abs() < 1e-6);

        // At r=1 (equal to mu), loss should be 0.5
        let l1 = loss.evaluate(1.0);
        assert!((l1 - 0.5).abs() < 1e-6);

        // At large r, loss should approach mu
        let l_large = loss.evaluate(1000.0);
        assert!(l_large < 1.5); // Approaches mu=1.0
    }

    #[test]
    fn test_geman_mcclure_loss_weight() {
        let loss = RobustLoss::GemanMcClure { mu: 1.0 };

        // At r near 0, weight should be high
        let w_small = loss.weight(0.1);
        assert!(w_small > 1.0);

        // At r=1 (equal to mu), weight should be 0.5
        let w1 = loss.weight(1.0);
        assert!((w1 - 0.5).abs() < 1e-6);

        // Weight should decrease with residual
        let w2 = loss.weight(2.0);
        assert!(w2 < w1);
    }

    #[test]
    fn test_welsch_loss_evaluate() {
        let loss = RobustLoss::Welsch { mu: 1.0 };

        // At r=0, loss should be 0
        let l0 = loss.evaluate(0.0);
        assert!((l0 - 0.0).abs() < 1e-6);

        // At large r, loss should approach mu
        let l_large = loss.evaluate(10.0);
        assert!((l_large - 1.0).abs() < 0.01); // Approaches mu=1.0
    }

    #[test]
    fn test_welsch_loss_weight() {
        let loss = RobustLoss::Welsch { mu: 1.0 };

        // At r=0, weight should be 1.0
        let w0 = loss.weight(0.0);
        assert!((w0 - 1.0).abs() < 1e-6);

        // Weight should decay exponentially with residual
        let w1 = loss.weight(1.0);
        assert!(w1 < 1.0);
        assert!(w1 > 0.3); // exp(-1) ≈ 0.368
    }

    #[test]
    fn test_huber_loss_evaluate() {
        let loss = RobustLoss::Huber { mu: 1.0 };

        // At r=0, loss should be 0
        let l0 = loss.evaluate(0.0);
        assert!((l0 - 0.0).abs() < 1e-6);

        // At r < mu, loss should be quadratic: 0.5*r^2
        let l_small = loss.evaluate(0.5);
        assert!((l_small - 0.125).abs() < 1e-6);

        // At r >= mu, loss should be linear
        let l_large = loss.evaluate(2.0);
        assert!((l_large - 1.5).abs() < 1e-6); // mu*(r - 0.5*mu) = 1.0*(2 - 0.5)
    }

    #[test]
    fn test_huber_loss_weight() {
        let loss = RobustLoss::Huber { mu: 1.0 };

        // At r <= mu, weight should be 1.0
        let w_small = loss.weight(0.5);
        assert!((w_small - 1.0).abs() < 1e-6);

        // At r > mu, weight should be mu/r
        let w_large = loss.weight(2.0);
        assert!((w_large - 0.5).abs() < 1e-6); // mu/r = 1/2
    }

    #[test]
    fn test_truncated_least_squares_loss() {
        let loss = RobustLoss::TruncatedLeastSquares { c: 1.0 };

        // At r < c, loss should be r^2
        let l_small = loss.evaluate(0.5);
        assert!((l_small - 0.25).abs() < 1e-6);

        // At r >= c, loss should be c^2
        let l_large = loss.evaluate(10.0);
        assert!((l_large - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_truncated_least_squares_weight() {
        let loss = RobustLoss::TruncatedLeastSquares { c: 1.0 };

        // At r < c, weight should be 1.0
        let w_small = loss.weight(0.5);
        assert!((w_small - 1.0).abs() < 1e-6);

        // At r >= c, weight should be 0.0 (outlier)
        let w_large = loss.weight(2.0);
        assert!((w_large - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cauchy_loss_evaluate() {
        let loss = RobustLoss::Cauchy { mu: 1.0 };

        // At r=0, loss should be 0
        let l0 = loss.evaluate(0.0);
        assert!((l0 - 0.0).abs() < 1e-6);

        // Loss should be positive for r > 0
        let l1 = loss.evaluate(1.0);
        assert!(l1 > 0.0);
    }

    #[test]
    fn test_cauchy_loss_weight() {
        let loss = RobustLoss::Cauchy { mu: 1.0 };

        // At r=0, weight should be 1.0
        let w0 = loss.weight(0.0);
        assert!((w0 - 1.0).abs() < 1e-6);

        // Weight should decrease with residual
        let w1 = loss.weight(1.0);
        assert!(w1 < 1.0);
    }

    #[test]
    fn test_gnc_optimizer_creation() {
        let opt_gm = GNCOptimizer::new_geman_mcclure(1.0);
        assert_eq!(opt_gm.max_iterations, 50);
        assert_eq!(opt_gm.gnc_iterations, 10);

        let opt_tls = GNCOptimizer::new_tls(1.0);
        assert_eq!(opt_tls.max_iterations, 50);

        let opt_welsch = GNCOptimizer::new_welsch(1.0);
        assert_eq!(opt_welsch.max_iterations, 50);
    }

    #[test]
    fn test_loss_param_update() {
        let mut loss = RobustLoss::GemanMcClure { mu: 1.0 };
        assert!((loss.get_param() - 1.0).abs() < 1e-6);

        loss.update_param(2.0);
        assert!((loss.get_param() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss_robustness_outlier_rejection() {
        // TLS with small threshold should heavily penalize outliers
        let loss_tls = RobustLoss::TruncatedLeastSquares { c: 1.0 };

        let inlier_loss = loss_tls.evaluate(0.5);
        let outlier_loss = loss_tls.evaluate(10.0);

        // Outlier loss should be bounded (c^2 = 1.0) despite large residual
        assert!(outlier_loss < inlier_loss * 10.0);
    }

    #[test]
    fn test_loss_smoothness_comparison() {
        let residuals = vec![0.0, 0.5, 1.0, 2.0, 5.0];
        let loss_gm = RobustLoss::GemanMcClure { mu: 1.0 };
        let loss_welsch = RobustLoss::Welsch { mu: 1.0 };

        let mut prev_gm = 0.0;
        let mut prev_welsch = 0.0;

        for &r in &residuals {
            let gm = loss_gm.evaluate(r);
            let welsch = loss_welsch.evaluate(r);

            // All losses should be monotonically increasing
            assert!(gm >= prev_gm);
            assert!(welsch >= prev_welsch);

            prev_gm = gm;
            prev_welsch = welsch;
        }
    }

    #[test]
    fn test_icp_point_clouds_small() {
        // Test with very small point clouds
        let source = PointCloud::new(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]);
        let target = PointCloud::new(vec![
            Point3::new(0.05, 0.0, 0.0),
            Point3::new(1.05, 0.0, 0.0),
            Point3::new(0.0, 1.05, 0.0),
        ]);

        let init = Matrix4::identity();
        let result = registration_icp_point_to_plane(&source, &target, 1.0, &init, 20);

        // Should handle small point clouds gracefully
        if let Some(res) = result {
            assert!(res.fitness >= 0.0);
            assert!(res.num_iterations > 0);
        }
    }
}
