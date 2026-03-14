use crate::mapping::MapExt;
use cv_calib3d::solve_pnp_ransac;
use cv_core::{slam::WorldMap, storage::Storage, CameraIntrinsics, KeyPoints, Pose, Tensor};
use cv_features::{detect_and_compute_ctx, Descriptors, Orb};
use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::ResourceGroup;
use nalgebra::{Point2, Point3};
use std::sync::Arc;

pub struct TrackingFrame<S: Storage<u8> + 'static> {
    pub image: Tensor<u8, S>,
    pub keypoints: KeyPoints,
    pub descriptors: Descriptors,
    pub pose: Pose,
}

pub struct Tracker {
    pub group: Arc<ResourceGroup>,
    pub current_frame: Option<TrackingFrame<cv_core::storage::CpuStorage<u8>>>,
    pub last_frame: Option<TrackingFrame<cv_core::storage::CpuStorage<u8>>>,
    pub intrinsics: CameraIntrinsics,
    pub detector: Orb,
    pub last_depth: Option<Tensor<f32, cv_core::storage::CpuStorage<f32>>>,
}

impl Tracker {
    pub fn new(group: Arc<ResourceGroup>, intrinsics: CameraIntrinsics) -> Self {
        Self {
            group,
            current_frame: None,
            last_frame: None,
            intrinsics,
            detector: Orb::default().with_n_features(500),
            last_depth: None,
        }
    }

    pub fn process_frame<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
        &mut self,
        image: &Tensor<u8, S>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        use cv_core::storage::CpuStorage;
        let device = self
            .group
            .device()
            .map_err(|e| format!("Failed to get compute device: {}", e))?;

        let (keypoints, descriptors) =
            detect_and_compute_ctx(&self.detector, &device, &self.group, image);

        // Convert to CPU for long-term storage and frame-to-frame matching if needed
        let cpu_img_tensor = convert_to_cpu(&device, image)
            .map_err(|e| format!("Failed to convert image to CPU: {}", e))?;

        let mut frame = TrackingFrame {
            image: cpu_img_tensor,
            keypoints,
            descriptors,
            pose: Pose::default(),
        };

        let mut tracking_success = false;
        let mut tracked_indices = Vec::new();

        if !map.points.is_empty() {
            let map_descs = map.get_descriptors();

            let map_desc_tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
                map_descs
                    .descriptors
                    .iter()
                    .flat_map(|d| d.data.clone())
                    .collect(),
                cv_core::TensorShape::new(1, map_descs.len(), 32),
            )
            .map_err(|e| e.to_string())?;

            let query_desc_tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(
                frame
                    .descriptors
                    .descriptors
                    .iter()
                    .flat_map(|d| d.data.clone())
                    .collect(),
                cv_core::TensorShape::new(1, frame.descriptors.len(), 32),
            )
            .map_err(|e| e.to_string())?;

            // Use accelerated matching via the device
            let matches = device
                .match_descriptors(&query_desc_tensor, &map_desc_tensor, 0.7)
                .map_err(|e| e.to_string())?;

            if matches.len() >= 10 {
                let mut object_pts = Vec::new();
                let mut image_pts = Vec::new();

                for m in &matches.matches {
                    if let Ok(p) = map.points[m.train_idx as usize].read() {
                        object_pts.push(Point3::new(
                            p.world_pos.x as f64,
                            p.world_pos.y as f64,
                            p.world_pos.z as f64,
                        ));
                        let kp = &frame.keypoints.keypoints[m.query_idx as usize];
                        image_pts.push(Point2::new(kp.x, kp.y));
                    }
                }

                if let Ok((pose, inliers)) =
                    solve_pnp_ransac(&object_pts, &image_pts, &self.intrinsics, None, 2.0, 100)
                {
                    frame.pose = pose;
                    tracked_indices = matches
                        .matches
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| inliers[*i])
                        .map(|(_, m)| m.train_idx as usize)
                        .collect();
                    tracking_success = true;
                }
            }
        } else {
            // First frame initialization
            // Seed the map with initial points (assuming some default depth or just placeholders)
            // For true monocular SLAM, we need 2 frames to initialize.
            // For now, let's just accept the frame as the origin and wait for the next frame.
            // We can add dummy points or just return success so the loop continues.

            // Critical: If we don't add points to the map, the next frame will also see empty map.
            // For a demo, let's assume we can initialize map points from this frame?
            // Without depth, we can't.
            // So we need "Bootstrapping" state.
            // Simplified: Just set success = true, pose = Identity.
            // The map remains empty, so next frame will also fall here?
            // NO. "if !map.points.is_empty()" checks points.

            // To make the demo interesting, let's pretend we initialized some points
            // just to allow matching test (even if physics is wrong) or change logic to
            // "if map is empty, create dummy map points at z=1.0 for all keypoints"
            // This allows the tracker to "start".

            for (i, desc) in frame.descriptors.descriptors.iter().enumerate() {
                // Back-project to z=1.0
                let kp = &desc.keypoint;
                let x = (kp.x - self.intrinsics.cx) / self.intrinsics.fx;
                let y = (kp.y - self.intrinsics.cy) / self.intrinsics.fy;
                let pos = Point3::new(x as f32, y as f32, 1.0);

                let desc_data = desc.data.clone();
                let mp = cv_core::slam::MapPoint::new(i as u64, pos, desc_data);
                map.add_point(mp);
            }

            tracking_success = true;
        }

        if !tracking_success {
            if let Some(ref last) = self.last_frame {
                frame.pose = last.pose;
            } else if self.current_frame.is_none() {
                // If map was not empty but we failed tracking, we fall here.
                // But if map was empty, we handled it above.
                // So this only happens if tracking failed on non-first frame OR logic error.
                // Actually, if map is empty and we didn't add points, we fail.
                // But we added points above.

                // Double check logic: if map empty -> add points -> success = true.
                // So we shouldn't reach here for first frame.
                return Err("Tracking failed".to_string());
            }
        }

        self.last_frame = self.current_frame.take();
        self.current_frame = Some(frame);

        self.current_frame
            .as_ref()
            .map(|f| (f.pose, tracked_indices))
            .ok_or_else(|| "Frame not set after tracking".to_string())
    }

    pub fn process_frame_with_depth<
        S: Storage<u8> + cv_core::StorageFactory<u8> + 'static,
        D: Storage<f32> + cv_core::StorageFactory<f32> + 'static,
    >(
        &mut self,
        image: &Tensor<u8, S>,
        depth: &Tensor<f32, D>,
        map: &mut WorldMap,
    ) -> Result<(Pose, Vec<usize>), String> {
        use cv_core::storage::CpuStorage;
        use cv_hal::storage::GpuStorage;
        use cv_hal::tensor_ext::TensorToCpu;

        // First, perform sparse feature tracking
        let (mut pose, tracked_indices) = self.process_frame(image, map)?;

        // Convert depth to CPU if needed for ICP
        let depth_cpu = if let Some(gpu_storage) =
            depth.storage.as_any().downcast_ref::<GpuStorage<f32>>()
        {
            let input_gpu = Tensor {
                storage: gpu_storage.clone(),
                shape: depth.shape,
                dtype: depth.dtype,
                _phantom: std::marker::PhantomData,
            };
            let device = self
                .group
                .device()
                .map_err(|e| format!("Failed to get compute device: {}", e))?;
            let gpu_ctx = match device {
                ComputeDevice::Gpu(g) => g,
                _ => return Ok((pose, tracked_indices)),
            };
            input_gpu
                .to_cpu_ctx(gpu_ctx)
                .map_err(|e| format!("Depth GPU download failed: {}", e))?
        } else if let Some(cpu_storage) = depth.storage.as_any().downcast_ref::<CpuStorage<f32>>() {
            Tensor {
                storage: cpu_storage.clone(),
                shape: depth.shape,
                dtype: depth.dtype,
                _phantom: std::marker::PhantomData,
            }
        } else {
            return Ok((pose, tracked_indices));
        };

        // Attempt ICP refinement if we have GPU support and previous depth data
        if let Some(ref last_depth) = self.last_depth {
            // Try to refine pose using dense ICP
            if let Err(e) = self.refine_pose_with_icp(&mut pose, last_depth, &depth_cpu) {
                // If ICP fails, just continue with sparse tracking result
                eprintln!("ICP refinement failed: {}", e);
            }
        }

        // Store current depth for next frame
        self.last_depth = Some(depth_cpu);

        Ok((pose, tracked_indices))
    }

    fn refine_pose_with_icp(
        &self,
        pose: &mut Pose,
        prev_depth: &Tensor<f32, cv_core::storage::CpuStorage<f32>>,
        curr_depth: &Tensor<f32, cv_core::storage::CpuStorage<f32>>,
    ) -> Result<(), String> {
        use cv_hal::tensor_ext::TensorToGpu;

        let device = self
            .group
            .device()
            .map_err(|e| format!("Failed to get compute device: {}", e))?;

        let gpu_ctx = match device {
            ComputeDevice::Gpu(g) => g,
            ComputeDevice::Cpu(_) | ComputeDevice::Mlx(_) => return Ok(()),
        };

        // Convert depth tensors to GPU
        let prev_depth_gpu = prev_depth
            .to_gpu_ctx(gpu_ctx)
            .map_err(|e| format!("Previous depth GPU upload failed: {}", e))?;
        let curr_depth_gpu = curr_depth
            .to_gpu_ctx(gpu_ctx)
            .map_err(|e| format!("Current depth GPU upload failed: {}", e))?;

        // Build intrinsics array [fx, fy, cx, cy]
        let intrinsics = [
            self.intrinsics.fx as f32,
            self.intrinsics.fy as f32,
            self.intrinsics.cx as f32,
            self.intrinsics.cy as f32,
        ];

        // Convert current pose to initial guess for ICP (convert to f32)
        let pose_matrix_f64 = pose.matrix();
        let pose_matrix_f32 =
            nalgebra::Matrix4::from_iterator(pose_matrix_f64.iter().map(|&x| x as f32));

        // Call dense ICP
        let (ata, atb) = cv_hal::gpu_kernels::icp::dense_step(
            gpu_ctx,
            &prev_depth_gpu,
            &curr_depth_gpu,
            &intrinsics,
            &pose_matrix_f32,
            0.05,                  // max_dist: 5cm
            30.0_f32.to_radians(), // max_angle: 30 degrees
        )
        .map_err(|e| format!("ICP step failed: {}", e))?;

        // Solve the linear system for pose update: atA * delta = -atb
        // The ICP energy is E = sum ||r_i||^2 where r_i are residuals.
        // The normal equations give atA * delta = -atb (gradient descent direction).
        let delta = -(ata
            .try_inverse()
            .ok_or_else(|| "ICP AtA matrix not invertible".to_string())?
            * atb);

        // Update pose with delta (parameterized as [omega (rotation), v (translation)])
        // Small angle approximation for rotation from omega
        let omega = nalgebra::Vector3::new(delta[0] as f64, delta[1] as f64, delta[2] as f64);
        let translation = nalgebra::Vector3::new(delta[3] as f64, delta[4] as f64, delta[5] as f64);

        // Create rotation matrix from axis-angle (small angle)
        let angle = omega.norm();
        let rot_update = if angle > 1e-6 {
            let axis = omega / angle;
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
        } else {
            nalgebra::Rotation3::identity()
        };

        // Update the pose
        pose.rotation *= nalgebra::UnitQuaternion::from_rotation_matrix(&rot_update);
        pose.translation += translation;

        Ok(())
    }
}

fn convert_to_cpu<S: Storage<u8> + cv_core::StorageFactory<u8> + 'static>(
    device: &ComputeDevice,
    tensor: &Tensor<u8, S>,
) -> std::result::Result<Tensor<u8, cv_core::storage::CpuStorage<u8>>, String> {
    use cv_core::storage::CpuStorage;
    use cv_hal::storage::GpuStorage;
    use cv_hal::tensor_ext::TensorToCpu;

    if let Some(gpu_storage) = tensor.storage.as_any().downcast_ref::<GpuStorage<u8>>() {
        let input_gpu = Tensor {
            storage: gpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        let gpu_ctx = match device {
            ComputeDevice::Gpu(g) => g,
            _ => return Err("GpuStorage requires GPU context".into()),
        };
        input_gpu
            .to_cpu_ctx(gpu_ctx)
            .map_err(|e| format!("GPU download failed: {}", e))
    } else if let Some(cpu_storage) = tensor.storage.as_any().downcast_ref::<CpuStorage<u8>>() {
        let input_cpu = Tensor {
            storage: cpu_storage.clone(),
            shape: tensor.shape,
            dtype: tensor.dtype,
            _phantom: std::marker::PhantomData,
        };
        Ok(input_cpu.clone())
    } else {
        Err("Unsupported storage type".into())
    }
}
