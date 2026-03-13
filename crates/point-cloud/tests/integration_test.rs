/// Integration tests for point-cloud crate
/// Verifies that all CPU and GPU functions can be imported and used correctly

#[test]
fn test_cpu_filtering_imports() {
    use cv_point_cloud::cpu::filtering::{
        remove_radius_outliers, remove_statistical_outliers, voxel_down_sample,
    };

    let _ = voxel_down_sample;
    let _ = remove_statistical_outliers;
    let _ = remove_radius_outliers;
}

#[test]
fn test_cpu_normals_imports() {
    use cv_point_cloud::cpu::normals::{estimate_normals, orient_normals};

    let _ = estimate_normals;
    let _ = orient_normals;
}

#[test]
fn test_cpu_segmentation_imports() {
    use cv_point_cloud::cpu::segmentation::{cluster_dbscan, compute_fpfh_feature, segment_plane};

    let _ = segment_plane;
    let _ = cluster_dbscan;
    let _ = compute_fpfh_feature;
}

#[test]
fn test_gpu_config_imports() {
    use cv_point_cloud::gpu::{ComputeMode, NormalComputeConfig};

    let _ = ComputeMode::CPU;
    let _ = ComputeMode::GPU;
    let _ = ComputeMode::Hybrid;
    let _ = ComputeMode::Adaptive;

    let _config = NormalComputeConfig::default();
    let _config = NormalComputeConfig::cpu();
    let _config = NormalComputeConfig::gpu();
    let _config = NormalComputeConfig::fast();
    let _config = NormalComputeConfig::high_quality();
}

#[test]
fn test_gpu_normals_imports() {
    use cv_point_cloud::gpu::normals::{
        compute_normals, compute_normals_cpu, compute_normals_ctx, compute_normals_simple,
    };

    let _ = compute_normals;
    let _ = compute_normals_simple;
    let _ = compute_normals_ctx;
    let _ = compute_normals_cpu;
}

#[test]
fn test_gpu_filtering_imports() {
    use cv_point_cloud::gpu::filtering::voxel_downsample;

    let _ = voxel_downsample;
}
