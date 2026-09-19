// TSDF Integration Shader (Voxel-centric)
// Each thread processes one voxel, projecting it into the depth map.

struct Params {
    voxel_size: f32,
    truncation_distance: f32,
    image_width: u32,
    image_height: u32,
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    padding: u32,
}

struct Voxel {
    tsdf: f32,
    weight: f32,
}

@group(0) @binding(0) var<storage, read> depth_image: array<f32>;
@group(0) @binding(1) var<storage, read> color_image: array<u32>; // Packed RGBA
@group(0) @binding(2) var<storage, read_write> voxels: array<Voxel>;
@group(0) @binding(3) var<storage, read_write> colors: array<u32>;

@group(1) @binding(0) var<uniform> world_to_camera: mat4x4<f32>;
@group(1) @binding(1) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(1) @binding(2) var<uniform> params: Params;

// Project 3D point in camera space to 2D pixel coordinates
fn project(p: vec3<f32>) -> vec2<f32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;
    
    let u = (p.x * fx / p.z) + cx;
    let v = (p.y * fy / p.z) + cy;
    
    return vec2<f32>(u, v);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vx = global_id.x;
    let vy = global_id.y;
    let vz = global_id.z;
    
    if (vx >= params.vol_x || vy >= params.vol_y || vz >= params.vol_z) {
        return;
    }
    
    // Voxel position in world space
    let p_world = vec3<f32>(
        (f32(vx) + 0.5) * params.voxel_size,
        (f32(vy) + 0.5) * params.voxel_size,
        (f32(vz) + 0.5) * params.voxel_size
    );
    
    // Transform to camera space
    let p_camera = (world_to_camera * vec4<f32>(p_world, 1.0)).xyz;
    
    if (p_camera.z <= 0.0) {
        return;
    }
    
    // Project to image
    let uv = project(p_camera);
    let u = u32(round(uv.x));
    let v = u32(round(uv.y));
    
    if (u >= params.image_width || v >= params.image_height) {
        return;
    }
    
    let pixel_idx = v * params.image_width + u;
    let depth = depth_image[pixel_idx];
    
    if (depth <= 0.0 || depth > 10.0) {
        return;
    }
    
    let dist = depth - p_camera.z;
    
    if (dist < -params.truncation_distance) {
        return;
    }
    
    let tsdf_sample = clamp(dist / params.truncation_distance, -1.0, 1.0);
    
    let idx = vz * params.vol_x * params.vol_y + vy * params.vol_x + vx;
    let old_v = voxels[idx];
    
    let weight = 1.0;
    let new_weight = min(old_v.weight + weight, 50.0);
    let new_tsdf = (old_v.tsdf * old_v.weight + tsdf_sample * weight) / new_weight;
    
    voxels[idx] = Voxel(new_tsdf, new_weight);
}
