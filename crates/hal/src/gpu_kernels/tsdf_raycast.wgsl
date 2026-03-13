// tsdf_raycast.wgsl
// GPU TSDF raycasting kernel for surface rendering

struct VolumeParams {
    origin: vec4<f32>,
    dim: vec4<u32>, // x, y, z, _
    voxel_size: f32,
    truncation: f32,
    _padding1: f32,
    _padding2: f32,
};

@group(0) @binding(0) var<storage, read> tsdf_volume: array<f32>; // (sdf, weight) interleaved or just sdf?
@group(0) @binding(1) var<storage, read_write> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: VolumeParams;

struct CameraParams {
    view_matrix: mat4x4<f32>,
    proj_matrix_inv: mat4x4<f32>,
    width: u32,
    height: u32,
};
@group(0) @binding(4) var<uniform> cam: CameraParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= cam.width || y >= cam.height) {
        return;
    }
    
    // 1. Calculate ray direction in world space
    let uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / vec2<f32>(f32(cam.width), f32(cam.height));
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0, 1.0);
    var ray_dir_cam = cam.proj_matrix_inv * ndc;
    ray_dir_cam = ray_dir_cam / ray_dir_cam.w;
    
    let ray_origin = (cam.view_matrix * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let ray_dir = normalize((cam.view_matrix * vec4<f32>(ray_dir_cam.xyz, 0.0)).xyz);
    
    // 2. Ray marching through the TSDF volume
    var t = 0.0;
    let max_t = 10.0; // 10 meters
    let step_size = params.voxel_size * 0.5;
    
    var prev_sdf = 1.0;
    
    for (var i = 0; i < 512; i = i + 1) {
        let p = ray_origin + ray_dir * t;
        let v_coords = (p - params.origin.xyz) / params.voxel_size;
        
        if (any(v_coords < vec3<f32>(0.0)) || any(v_coords >= vec3<f32>(params.dim.xyz))) {
            t = t + step_size;
            continue;
        }
        
        let vi = vec3<u32>(v_coords);
        let idx = vi.z * params.dim.x * params.dim.y + vi.y * params.dim.x + vi.x;
        let sdf = tsdf_volume[idx * 2u]; // Assuming (sdf, weight)
        
        // Zero-crossing check
        if (prev_sdf > 0.0 && sdf < 0.0) {
            // Refine intersection with linear interpolation
            let refined_t = t - step_size * sdf / (sdf - prev_sdf);
            let hit_pt = ray_origin + ray_dir * refined_t;
            let out_idx = y * cam.width + x;
            vertices[out_idx] = vec4<f32>(hit_pt, 1.0);
            
            // Central difference for normal
            // (Simplified normal estimation)
            normals[out_idx] = vec4<f32>(-ray_dir, 0.0); 
            return;
        }
        
        prev_sdf = sdf;
        t = t + step_size;
        if (t > max_t) { break; }
    }
    
    let out_idx = y * cam.width + x;
    vertices[out_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
