// TSDF Raycasting Shader
// Casts rays into the volume to render the zero-level set.

struct Params {
    width: u32,
    height: u32,
    voxel_size: f32,
    truncation: f32,
    step_factor: f32,
    min_depth: f32,
    max_depth: f32,
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
}

struct Voxel {
    tsdf: f32,
    weight: f32,
}

@group(0) @binding(0) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(1) var<storage, read_write> output_data: array<vec4<f32>>; // x: depth, yzw: normal
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> camera_to_world: mat4x4<f32>;
@group(0) @binding(4) var<uniform> intrinsics_inv: vec4<f32>; // 1/fx, 1/fy, cx, cy

fn get_tsdf(p: vec3<f32>) -> f32 {
    let x = i32(floor(p.x / params.voxel_size));
    let y = i32(floor(p.y / params.voxel_size));
    let z = i32(floor(p.z / params.voxel_size));

    if (x < 0 || x >= i32(params.vol_x) || y < 0 || y >= i32(params.vol_y) || z < 0 || z >= i32(params.vol_z)) {
        return 0.0;
    }

    let idx = u32(z) * params.vol_x * params.vol_y + u32(y) * params.vol_x + u32(x);
    return voxels[idx].tsdf;
}

fn get_tsdf_interp(p: vec3<f32>) -> f32 {
    return get_tsdf(p);
}

fn compute_normal(p: vec3<f32>) -> vec3<f32> {
    let d = params.voxel_size;
    let dx = get_tsdf_interp(p + vec3<f32>(d, 0.0, 0.0)) - get_tsdf_interp(p - vec3<f32>(d, 0.0, 0.0));
    let dy = get_tsdf_interp(p + vec3<f32>(0.0, d, 0.0)) - get_tsdf_interp(p - vec3<f32>(0.0, d, 0.0));
    let dz = get_tsdf_interp(p + vec3<f32>(0.0, 0.0, d)) - get_tsdf_interp(p - vec3<f32>(0.0, 0.0, d));
    
    let n = vec3<f32>(dx, dy, dz);
    if (length(n) > 0.0) {
        return normalize(n);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let u = f32(x);
    let v = f32(y);
    
    let mx = (u - intrinsics_inv.z) * intrinsics_inv.x;
    let my = (v - intrinsics_inv.w) * intrinsics_inv.y;
    
    let dir_cam = normalize(vec3<f32>(mx, my, 1.0));
    let origin_cam = vec3<f32>(0.0, 0.0, 0.0);

    let dir_world = (camera_to_world * vec4<f32>(dir_cam, 0.0)).xyz;
    let origin_world = (camera_to_world * vec4<f32>(origin_cam, 1.0)).xyz;

    var t = params.min_depth;
    var found = false;
    var t_step = params.voxel_size * params.step_factor;

    var prev_tsdf = get_tsdf(origin_world + dir_world * t);
    
    loop {
        if (t >= params.max_depth) { break; }
        
        let p = origin_world + dir_world * t;
        let tsdf_val = get_tsdf(p);
        
        if (prev_tsdf > 0.0 && tsdf_val < 0.0 && tsdf_val > -0.8) {
            let t_surf = t - t_step * (tsdf_val / (tsdf_val - prev_tsdf));
            let p_surf_world = origin_world + dir_world * t_surf;
            
            // Convert p_surf_world back to camera frame to get Z depth
            // Actually, we can just use t_surf * dir_cam.z if we had dir_cam.
            // But dir_cam was normalized. dir_cam.z = 1.0 / length(vec3(mx, my, 1.0))
            let z_depth = t_surf * (1.0 / length(vec3<f32>(mx, my, 1.0)));

            let normal = compute_normal(p_surf_world);
            output_data[y * params.width + x] = vec4<f32>(z_depth, normal.x, normal.y, normal.z);
            
            found = true;
            break;
        }
        
        prev_tsdf = tsdf_val;
        t += t_step;
    }

    if (!found) {
        output_data[y * params.width + x] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
