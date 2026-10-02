// Ray-Mesh Intersection Shader
// Möller-Trumbore ray-triangle intersection for batch ray casting
//
// Storage-buffer budget: this device requests
// `wgpu::Limits::downlevel_defaults()`, which caps
// `max_storage_buffers_per_shader_stage` at 4, and wgpu-core derives the bind
// group layout per ENTRY POINT from the bindings that entry point reaches. This
// file previously declared seven `@group(0)` storage buffers, all of them
// reached by `main`, so `create_compute_pipeline` failed outright with
// "Too many bindings of type StorageBuffers, limit is 4".
//
// Two pairs are packed into single buffers to get to four:
//   * `ray_origins` + `ray_directions` -> `rays`, interleaved as
//     `struct Ray { origin: vec3<f32>, dir: vec3<f32> }` (32-byte stride).
//   * `hit_points` + `hit_normals` + `hit_distances` -> `hits`, interleaved as
//     `struct Hit { point: vec3<f32>, normal: vec3<f32>, dist: f32, _pad: f32 }`
//     (48-byte stride).
//
// The 16-byte strides of `vec3<f32>` are unchanged: `origin`/`dir` sit at byte
// 0 and 16 within a 32-byte `Ray`, and `point`/`normal` at byte 0 and 16 within
// a 48-byte `Hit`, so the host's `GpuVec3 { [f32; 3], padding: f32 }` packing
// still describes them exactly.

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
};

// `Hit` packs the distance into `w` and pads the normal with a fourth float so
// the record is two whole vec4s. A struct of three `vec3<f32>` plus an `f32`
// would not line up: WGSL places the trailing `f32` at byte 28, not 32, and
// rounds the struct up to 48, so the distance would not sit where a `#[repr(C)]`
// host struct puts it. Two `vec4`s give offsets 0 and 16 and a 32-byte stride
// with no padding on either side.
struct Hit {
    point_dist: vec4<f32>, // xyz = hit point, w = distance (-1 when no hit)
    normal_pad: vec4<f32>, // xyz = hit normal, w unused
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> mesh_vertices: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> mesh_faces: array<vec3<u32>>; // Triangle indices
@group(0) @binding(3) var<storage, read_write> hits: array<Hit>;

struct RaycastParams {
    num_rays: u32,
    num_faces: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(1) @binding(0) var<uniform> params: RaycastParams;

const EPSILON: f32 = 0.00001;

// Möller-Trumbore ray-triangle intersection
fn ray_triangle_intersect(
    orig: vec3<f32>,
    dir: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
) -> vec4<f32> { // Returns (t, u, v, hit) where hit=1 if intersected
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    
    if (abs(a) < EPSILON) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Parallel
    }
    
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    
    if (v < 0.0 || u + v > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let t = f * dot(edge2, q);
    
    if (t > EPSILON) {
        return vec4<f32>(t, u, v, 1.0);
    }
    
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    
    if (ray_idx >= params.num_rays) {
        return;
    }
    
    let ray = rays[ray_idx];
    let orig = ray.origin;
    let dir = normalize(ray.dir);
    
    var closest_t = 999999.0;
    var hit_found = false;
    var hit_u = 0.0;
    var hit_v = 0.0;
    var hit_face_idx = 0u;
    
    // Test against all triangles
    for (var face_idx = 0u; face_idx < params.num_faces; face_idx = face_idx + 1u) {
        let face = mesh_faces[face_idx];
        let v0 = mesh_vertices[face.x];
        let v1 = mesh_vertices[face.y];
        let v2 = mesh_vertices[face.z];
        
        let result = ray_triangle_intersect(orig, dir, v0, v1, v2);
        
        if (result.w > 0.0 && result.x < closest_t) {
            closest_t = result.x;
            hit_u = result.y;
            hit_v = result.z;
            hit_face_idx = face_idx;
            hit_found = true;
        }
    }
    
    if (hit_found) {
        let hit_point = orig + dir * closest_t;
        let face = mesh_faces[hit_face_idx];
        let v0 = mesh_vertices[face.x];
        let v1 = mesh_vertices[face.y];
        let v2 = mesh_vertices[face.z];
        let normal = normalize(cross(v1 - v0, v2 - v0));
        
        hits[ray_idx].point_dist = vec4<f32>(hit_point, closest_t);
        hits[ray_idx].normal_pad = vec4<f32>(normal, 0.0);
    } else {
        hits[ray_idx].point_dist = vec4<f32>(0.0, 0.0, 0.0, -1.0); // w = no hit
        hits[ray_idx].normal_pad = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}