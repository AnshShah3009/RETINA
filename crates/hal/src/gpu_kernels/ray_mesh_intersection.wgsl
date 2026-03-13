// Ray-Mesh Intersection Shader
// Möller-Trumbore ray-triangle intersection for batch ray casting

@group(0) @binding(0) var<storage, read> ray_origins: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> ray_directions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> mesh_vertices: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> mesh_faces: array<vec3<u32>>; // Triangle indices
@group(0) @binding(4) var<storage, read_write> hit_distances: array<f32>;
@group(0) @binding(5) var<storage, read_write> hit_points: array<vec3<f32>>;
@group(0) @binding(6) var<storage, read_write> hit_normals: array<vec3<f32>>;

@group(1) @binding(0) var<uniform> num_rays: u32;
@group(1) @binding(1) var<uniform> num_faces: u32;

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
    
    if (ray_idx >= num_rays) {
        return;
    }
    
    let orig = ray_origins[ray_idx];
    let dir = normalize(ray_directions[ray_idx]);
    
    var closest_t = 999999.0;
    var hit_found = false;
    var hit_u = 0.0;
    var hit_v = 0.0;
    var hit_face_idx = 0u;
    
    // Test against all triangles
    for (var face_idx = 0u; face_idx < num_faces; face_idx = face_idx + 1u) {
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
        
        hit_distances[ray_idx] = closest_t;
        hit_points[ray_idx] = hit_point;
        hit_normals[ray_idx] = normal;
    } else {
        hit_distances[ray_idx] = -1.0; // No hit
    }
}
