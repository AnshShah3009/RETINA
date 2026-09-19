// Voxel Grid Downsample Shader
// Performs voxel grid downsampling on GPU

@group(0) @binding(0) var<storage, read> input_points: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> input_colors: array<u32>; // Optional
@group(0) @binding(2) var<storage, read_write> voxel_hash_keys: array<u32>; // Spatial hash
@group(0) @binding(3) var<storage, read_write> voxel_indices: array<u32>; // Point indices per voxel
@group(0) @binding(4) var<storage, read_write> voxel_counts: array<u32>; // Points per voxel

@group(1) @binding(0) var<uniform> num_points: u32;
@group(1) @binding(1) var<uniform> voxel_size: f32;
@group(1) @binding(2) var<uniform> grid_origin: vec3<f32>;
@group(1) @binding(3) var<uniform> grid_dims: vec3<u32>; // Grid dimensions for hashing

// Spatial hash function
fn spatial_hash(point: vec3<f32>) -> u32 {
    let voxel_coord = vec3<i32>((point - grid_origin) / voxel_size);
    
    // Simple 3D to 1D hash
    let x = u32(voxel_coord.x);
    let y = u32(voxel_coord.y);
    let z = u32(voxel_coord.z);
    
    // Hash with prime numbers
    return (x * 73856093u + y * 19349663u + z * 83492791u) % (num_points * 2u);
}

@compute @workgroup_size(256)
fn compute_hashes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= num_points) {
        return;
    }
    
    let point = input_points[idx];
    let hash = spatial_hash(point);
    
    voxel_hash_keys[idx] = hash;
    voxel_indices[idx] = idx;
}

// Note: Sorting and counting unique voxels would be done in separate passes
// using radix sort and parallel prefix sum
